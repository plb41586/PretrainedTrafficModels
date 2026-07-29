# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Dissertation codebase for pretraining raw-byte traffic models (packet/flow-level encoders,
autoencoders, MLM) and comparing backbones (Mamba vs. Transformer) for network anomaly
detection. It bundles two mostly-independent subsystems:

- `RawByteTrafficModelling/` — Python/PyTorch model code, pretraining scripts, and anomaly
  detection experiments.
- `feature_extraction/` — a Rust CLI (`pcap-analyzer`) that parses PCAP files into flow
  features, optionally pushing them to FalkorDB (graph DB) and Redis, or dumping to Polars.

There is a Python client (`feature_extraction/pythonclient/client.py`) that reads the
msgpack-serialized payload sets the Rust side writes to Redis.

## Environment

This project is meant to run inside the devcontainer defined in `.devcontainer/`
(`pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` base image). Key environment facts baked into
the container:
- `PYTHONPATH` includes `/workspace` (the repo root), which is why Python modules use
  absolute imports like `from RawByteTrafficModelling.ModelComponents.ModelDefinitions import ...`
  — run scripts as modules from the repo root (`python -m RawByteTrafficModelling.PreTraining.PacketLevelMLM`),
  not as bare scripts, or the imports will fail.
- A venv at `/home/<user>/app/venv` has GPU-only deps preinstalled (`mamba-ssm`, `causal-conv1d`,
  `keras-nlp`/`keras_hub`, `torch`) that are not all captured in `requirements.txt`.
- Rust toolchain via rustup; `feature_extraction/target/release` is on `PATH`.

There is no test suite (no `pytest`/`unittest` files anywhere in the repo) and no lint/CI
config. Don't invent test or lint commands — verify changes by running the relevant script or
`cargo build`/`cargo check`.

## Common commands

Python (run from repo root so `RawByteTrafficModelling` resolves as a package):
```
python -m RawByteTrafficModelling.PreTraining.PacketLevelMLM
python -m RawByteTrafficModelling.PreTraining.PacketLevelAutoEncoder
python -m RawByteTrafficModelling.AnomalyDetection.AutoEncoderAD
```
These are not CLI tools — they are scripts with hardcoded config (paths, hyperparameters,
`output_dir`, GPU device) at the top of the file. To change training config, edit the
constants directly rather than adding argparse.

Rust (`feature_extraction/`):
```
cargo build --release            # from feature_extraction/
cargo run --release -- --file <pcap> [--cache-payloads] [--graph-name <name>] [--pl-outfile <path>]
```

Data splitting utility:
```
python -m data_tools.SplitDataDF   # edit DATA_FILE/OUTPUT_DIR constants at bottom of file first
```

## Architecture: model components

Everything lives under `RawByteTrafficModelling/ModelComponents/`:

- **`BackBones.py`** — swappable sequence backbones behind a common `(batch, seq_len, dim) ->
  (batch, seq_len, dim)` interface: `MambaBackbone` (stacked `mamba_ssm.Mamba` blocks) and
  `TransformerBackbone` (sinusoidal positional encoding + `nn.TransformerEncoder`, optional
  causal mask). Each has a matching `@dataclass` params type (`MambaBackboneParams`,
  `TransformerBackboneParams`) inheriting from `BackboneParams`.
- **`ModelDefinitions.py`** — the model zoo, built around a **backbone factory** pattern:
  `BACKBONES = {"Transformer": ..., "Mamba": ...}` and `build_backbone(kind, params)` /
  `unpack_backbone_params(type, config)` construct/deserialize whichever backbone a config
  names. Every model level (packet encoder, packet decoder, sequence encoder, sequence
  decoder) is backbone-agnostic and takes its concrete backbone as an injectable arg, falling
  back to `build_backbone` when none is passed.
- **`DataUtils.py`** — `ID_Encoder` turns raw byte arrays into fixed-length (1520) token-ID
  sequences with a CLS token placed either at SOS or EOS, plus pad/end-of-sequence tokens.
  `TrainingDatasetHandler` / `ValidationDatasetHandler` / `PreTrainingDatasetHandler` wrap a
  Polars DataFrame (columns include `data`, `mask`, `AttackLabel`, `FlowID`,
  `proto_hierarchy`) and provide batch/sequence sampling (by label, by flow, or by raw epoch
  index).
- **`FlowID.py`** — derives a `FlowID` for each packet row by hashing sorted sender/receiver
  address bytes extracted via the packet's byte mask (works for both Ethernet-only and
  IPv4/IPv6-bearing packets); `add_Flow_ID` appends this as a Polars column.

### Model hierarchy (packet level → sequence level)

There are three nested levels of encoder/decoder, each with its own params dataclass and a
`load_*_checkpoint` / `unpack_*_params` pair (checkpoints store a `dataclasses.asdict()` of
the params alongside `model_state_dict`, see `save_checkpoint`):

1. **Packet level** (`EncoderParams`, `AutoEncoderParams`, `MLM_Params`,
   `PacketClassifierParams`):
   - `Packet_Encoder` — embedding → backbone → `DynamicCLSPooling` (finds the CLS token
     position per-sequence dynamically from token IDs, rather than assuming a fixed index) →
     single latent vector per packet.
   - `Packet_MLM` — encoder + a per-token reconstruction head + a CLS classification head
     (used for masked-token pretraining with an auxiliary protocol-hierarchy classification
     loss).
   - `AutoregressiveDecoder` / `PacketAutoencoder` — decodes a packet latent back into bytes
     autoregressively (teacher forcing in `forward`, sampling in `generate`); the latent is
     projected and prepended as position 0 of the decoder input.
   - `Packet_Classifier` — encoder + linear head for supervised packet-level classification.

2. **Sequence level** (`SeqEncoderParams`, `SeqAutoEncoderParams`): `Sequence_Encoder` treats
   a (usually frozen, see `freeze_packet_encoder`/`unfreeze_packet_encoder`) `Packet_Encoder`
   as an "embedding layer" for whole packets, runs a second backbone over the per-packet
   latents, and pools a CLS vector written at index `seq_len` (right after the last real
   packet) — this keeps pooling valid even under a causal backbone. `SequenceDecoder`
   reconstructs all packet latents in parallel from that one flow vector using learned
   positional query tokens. `SequenceAutoencoder` wires encoder+decoder together, normalizes
   targets (`set_target_stats`/`normalize`/`denormalize`, since packet latents aren't
   zero-mean/unit-std), and has an optional length-prediction auxiliary head/loss so the
   bottleneck vector alone is enough to know how many packets to reconstruct.
   `precompute_latents` caches frozen packet-encoder outputs so sequence-AE training doesn't
   redo the packet forward pass every step.

3. **Legacy/alternate sequence classifier**: `SequenceClassifier` — an older, self-contained
   Mamba-based sequence classifier (not built via the generic backbone factory) with both a
   Python-loop `forward` and a vectorized `forward_ff` variant kept side by side for
   comparison.

Evaluation helpers at the bottom of `ModelDefinitions.py` (`retrieval_accuracy`,
`baseline_mses`, `byte_level_reconstruction`) are all `@torch.no_grad()` and operate on the
padding-masked, normalized latents produced by `SequenceAutoencoder.forward` — use
`build_padding_mask(seq_lens, num_packets)` rather than re-deriving padding masks elsewhere.

### Working with this hierarchy

- When adding a new backbone, add both a params dataclass (in `BackBones.py`) and register it
  in the `BACKBONES` dict and the `unpack_backbone_params` dispatcher in `ModelDefinitions.py`
  — both must stay in sync or checkpoint loading will silently fail to reconstruct configs.
- Config dataclasses nest raw dicts for their sub-configs (e.g. `AutoEncoderParams.DecBackbone`
  is a plain dict on disk); always go through the corresponding `unpack_*_params` function
  after `torch.load`, never construct the dataclass directly from a checkpoint's `config` dict.
- `RawByteTrafficModelling/PreTraining/SequenceLevelAutoEncoder.py` is a work-in-progress
  sketch (references an undefined `loader`, uses stale `d_model=` kwargs where
  `BackboneParams` now expects `dim=`) — treat it as a draft to finish, not a working
  reference, when asked to build on sequence-level AE training.

## Data conventions

- Packet byte sequences are fixed at **1520 tokens** (`packet_id_len`); vocab size is
  typically **262** = 256 byte values + special tokens (`<pad>=256`, `</s>=257`, `<CLS>=258`,
  `<mask>=259`, `<EndPointMasking>=260`, `<BOS>=261` — defined ad hoc per-script, not centrally).
- Datasets are Polars DataFrames (parquet under `data_artefacts/`, gitignored) with at least
  `data` (raw bytes), `mask` (which bytes to redact for MLM/PII), `AttackLabel`, `FlowID`, and
  `proto_hierarchy` columns.
- `data_artefacts/`, model checkpoints (`*.pth`, `*.ckpt`), and logs are gitignored — training
  scripts write into per-run subdirectories under `RawByteTrafficModelling/PreTraining/TrainingOutputs/`
  or `.../AnomalyDetection/Outputs/` that must exist before the script's `logging.FileHandler`
  is created (scripts do not `mkdir` their output dir).

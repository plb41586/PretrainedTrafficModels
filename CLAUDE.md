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

## Running commands (read before executing anything)

**Nothing in this project runs on the host.** The host has no usable interpreter and no GPU
deps — every Python invocation, script run, smoke test, `cargo` build, and data-tool run
happens inside the devcontainer. A host `python`/`python3`/`py_compile` call is always wrong;
it will fail with `No such file or directory` or, worse, pick up a stray interpreter and
report a misleading result.

**Ask before every single container command.** Do not batch them, do not treat one approval
as covering the next, and never fire one off "just to check something quickly". For each
command, before running it:

1. Show the **entire** command verbatim — no abbreviation, no `...`.
2. Give a one- or two-sentence reason why it is the right thing to run now.
3. Wait for explicit confirmation.

Reading files, `grep`, `ls`, and `git` on the host need no confirmation — the restriction is
about *executing* project code.

The container is a long-running `docker compose` service named **`devcontainer`** (user
`plb41586`, uid 1000). The repo is bind-mounted from the host at
`/home/plb41586/projects/PretrainedTrafficModels` to `/home/plb41586/workspace` inside it, so
edits made on the host are already visible — there is nothing to copy or rebuild. The venv is
on `PATH` via a Dockerfile `ENV`, so plain `python` is the venv's python even in a
non-interactive `docker exec`.

Canonical form (the `-w` is required; module imports resolve via cwd):
```
docker exec -w /home/plb41586/workspace devcontainer python -m RawByteTrafficModelling.PreTraining.SequenceLevelAutoEncoder
```
Add `bash -lc '<command>'` only when shell features are needed (pipes, redirection, globs) —
the container shell is bash, the host shell is fish, so quoting differs. Prefer letting the
user launch long training runs themselves with the `! <command>` prefix; propose the exact
command rather than running it.

Verification etiquette: since a syntax check is itself a container command, expect to hand
edits over unverified. Say plainly what has not been run, and give the exact command that
would check it. Prefer one meaningful run (a real script with a small
`MAX_STEPS_PER_EPOCH`) over a series of small probes.

## Environment

This project runs inside the devcontainer defined in `.devcontainer/`
(`pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel` base image, compose service and container name
both `devcontainer`). Key environment facts baked into the container:
- The workspace folder is `/home/plb41586/workspace` (bind mount of the repo root).
- Python modules use absolute imports like
  `from RawByteTrafficModelling.ModelComponents.ModelDefinitions import ...`, so run scripts as
  modules **from the workspace folder** (`python -m RawByteTrafficModelling.PreTraining.PacketLevelMLM`),
  never as bare scripts. What puts the repo on `sys.path` is cwd, not `PYTHONPATH`: the
  `.bashrc` line exports `/workspace`, which is a leftover from an older mount point and does
  not exist in the container. Hence `-w /home/plb41586/workspace` on every `docker exec`.
- A venv at `/home/plb41586/app/venv` has GPU-only deps preinstalled (`mamba-ssm`,
  `causal-conv1d`, `keras-nlp`/`keras_hub`, `torch`) that are not all captured in
  `requirements.txt`. It is first on `PATH` via `ENV`, so `python` is the venv's python without
  activating anything.
- **Two** RTX A5000s (24 GB each) are attached to the container and both are usable. Independent
  runs take one device each and go concurrently — this beats DDP here, because the data pipeline
  is synchronous single-threaded CPU work on the main thread and DDP would not shrink that
  serial fraction. Select with `RunConfig.resolve_device(index)`; scripts carry a `DEVICE_INDEX`
  constant. Anything importing `keras_hub` must pin TensorFlow to the CPU *before* that import
  (`import tensorflow as tf; tf.config.set_visible_devices([], "GPU")`), or TF preallocates
  nearly all memory on **both** GPUs and starves the other run.
- `redis` runs as a second service on the `devnet` network, reachable at host `redis`.
- Rust toolchain via rustup; `feature_extraction/target/release` is on `PATH` (via the same
  stale `/workspace` prefix — build and invoke by path if that bites).

There is no test suite (no `pytest`/`unittest` files anywhere in the repo) and no lint/CI
config. Don't invent test or lint commands — verify changes by running the relevant script or
`cargo build`/`cargo check`, in the container, after asking (see "Running commands" above).

## Common commands

All of these run **inside the container**, and each one needs its own confirmation first
(see "Running commands"). They are written below in bare form for readability; the actual
invocation is always `docker exec -w /home/plb41586/workspace devcontainer <command>`.

Python (cwd must be the workspace folder so `RawByteTrafficModelling` resolves as a package):
```
python -m RawByteTrafficModelling.PreTraining.PacketLevelMLM
python -m RawByteTrafficModelling.PreTraining.PacketLevelAutoEncoder
python -m RawByteTrafficModelling.PreTraining.SequenceLevelAutoEncoder
python -m RawByteTrafficModelling.PreTraining.CachePacketLatents   # both splits in one run
python -m RawByteTrafficModelling.PreTraining.CheckLatentCache     # read-only cache verification
python -m RawByteTrafficModelling.AnomalyDetection.AutoEncoderAD
```
These are not CLI tools — they are scripts with hardcoded config (paths, hyperparameters,
`output_dir`, GPU device) at the top of the file. To change training config, edit the
constants directly rather than adding argparse.

What is *not* per-experiment lives in `RawByteTrafficModelling/PreTraining/RunConfig.py`:
the dataset path registry (`DATASETS`), `SPECIAL_IDS` / `VOCAB_SIZE` / `PACKET_ID_LEN`,
`make_id_encoder()`, `packet_encoder_params()`, `setup_run()` (mkdir + logging),
`resolve_device()`, `MetricsCsv`, `cosine_warmup_lambda()`, `fixed_eval_batches()` and
`plot_curves()`. New scripts should import these rather than re-deriving them; hyperparameters
still belong as constants in the script. `PacketLevelAutoEncoder.py` is the current template.

Rust (`feature_extraction/`):
```
cargo build --release            # from feature_extraction/
cargo test                       # 15 unit tests, no database needed
cargo run --release -- (--file <pcap> | --interface <name>) \
    [--pl-outfile <path>] [--pl-chunk-size <n>] [--limit <n>] \
    [--graph-name <name>] [--topology <path>] [--falkor-host <host:port>] \
    [--cache-payloads] [--redis-host <host:port>]
```
`cargo` lives at `~/.cargo/bin` and is only on `PATH` through the login shell, so container
invocations need `bash -lc 'cargo ...'`.

**Every backend is opt-in.** With no `--graph-name`, FalkorDB is never contacted; with no
`--cache-payloads`, Redis is never contacted; with no `--pl-outfile`, nothing is exported.
Omitting both `--file` and `--interface` exits 2 — there is no default input path. This repo
only uses the Parquet path (there is no FalkorDB service in the compose file and no topology
TOML), so the usual invocation is just `--file <pcap> --pl-outfile <path>`. `--limit N` stops
after N packets and is the cheapest smoke test. See `feature_extraction/MIGRATION.md` for the
full pre-merge → merged migration notes.

Data splitting utility (config from constants at the bottom of the file):
```
python -m data_tools.SplitFlowsDF   # flow-level: whole flows per split, long flows cut by time
```
There is only one splitter. `SplitDataDF` — packet-level contiguous row slices that ignored
flows, and so tore a single flow across train/test/val — was deleted for leakage; do not
reintroduce a packet-level split, and treat `flow_split/` as the only valid partition.

`SplitFlowsDF` keeps every flow in a single split, cutting only long-lived flows
chronologically (train -> test -> val), and hits the ratios in packets. It assumes an
attack-free capture (see its module docstring). It groups on a canonical conversation key
derived in Python, which was a workaround for the pre-merge extractor writing a directional
`flow_key` fragmented by `proto_hierarchy`. The merged extractor writes a normalized key
carrying only the transport token, so on current parquets the canonicalization is a redundant
no-op mapping each `flow_key` to itself. Nothing still reads the pre-merge parquets, so it can
be collapsed to a plain `group_by("flow_key")` whenever `data_artefacts/deprecated_*` is
deleted (see `TODO.md`). The `flow_key` column is written out unchanged either way, so the
latent-cache path is unaffected.

**Do not mix key formats within one artefact lineage.** Re-extracting a capture changes its
`flow_key` strings, so its split and any latent cache keyed on them
(`flow_split/latents_*`) must be regenerated together — the `flow_key` asserts in
`CheckLatentCache.py` and `SequenceLevelAutoEncoder.py` are what catch a mismatch.

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
  `PreTrainingDatasetHandler` is the one in live use: it wraps a Polars DataFrame
  (`data`, `mask`, `proto_hierarchy`, plus `flow_key`/`timestamp_*` for `build_flow_index`) and
  provides sampling by flow or by raw epoch index. `TrainingDatasetHandler` /
  `ValidationDatasetHandler` sample by label and are **dead code** — they need `AttackLabel` and
  `FlowID` columns that no current parquet has. Also here: `load_latent_cache` (verifies the
  cache's checkpoint sha256) and `CachedLatentSequenceHandler` (windowed batches over cached
  packet latents).
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

Evaluation helpers at the bottom of `ModelDefinitions.py` are all `@torch.no_grad()`.
`baseline_mses` operates on the padding-masked, normalized latents produced by
`SequenceAutoencoder.forward`; `byte_level_reconstruction` / `decoder_byte_accuracy` score
teacher-forced byte accuracy through a frozen packet decoder (the same way
`PacketLevelAutoEncoder` scores itself, so the levels are comparable) and report `all` /
`nonpad` — the all-position number reads high on predicting `<pad>` alone. Use
`build_padding_mask(seq_lens, num_packets)` rather than re-deriving padding masks elsewhere.

A `retrieval_accuracy` helper (nearest-neighbour identification among the batch's true
latents) was removed on purpose: in this traffic, near-duplicate packets are pervasive and
the candidate pool is ~9.6k per val batch, so it sat near chance while reconstruction MSE
improved 16x — it measured the density of the target space, not the model. Byte accuracy
against the packet-AE ceiling replaced it; don't reintroduce it.

### Working with this hierarchy

- When adding a new backbone, add both a params dataclass (in `BackBones.py`) and register it
  in the `BACKBONES` dict and the `unpack_backbone_params` dispatcher in `ModelDefinitions.py`
  — both must stay in sync or checkpoint loading will silently fail to reconstruct configs.
- Config dataclasses nest raw dicts for their sub-configs (e.g. `AutoEncoderParams.DecBackbone`
  is a plain dict on disk); always go through the corresponding `unpack_*_params` function
  after `torch.load`, never construct the dataclass directly from a checkpoint's `config` dict.
- `RawByteTrafficModelling/PreTraining/SequenceLevelAutoEncoder.py` is finished and has trained
  end to end; the old "work-in-progress sketch" warning no longer applies. It is the reference
  for how a training script here should look (warmup+cosine, resume, metrics CSV, curves) —
  though it predates `RunConfig.py` and still carries its own copies of those helpers.

## Data conventions

- Packet byte sequences are fixed at **1520 tokens** (`packet_id_len`); vocab size is **262** =
  256 byte values + special tokens (`<pad>=256`, `</s>=257`, `<CLS>=258`, `<mask>=259`,
  `<EndPointMasking>=260`, `<BOS>=261`). These now live centrally in
  `RawByteTrafficModelling/PreTraining/RunConfig.py` (`SPECIAL_IDS`, `VOCAB_SIZE`,
  `PACKET_ID_LEN`); the older scripts still carry ad-hoc copies, so prefer the module.
- The `ID_Encoder` CLS placement is `"EOS"` and is load-bearing across levels — a latent cache
  built one way cannot be read by a model expecting the other. Go through
  `RunConfig.make_id_encoder()` rather than constructing one inline.
- **Where the data lives:** `data_artefacts/merged_extractor/<capture>/` with
  `NormalMerged.parquet`, `attacks/*.parquet` and `flow_split/{train,test,val}.parquet` plus
  `split_report.json`. Latent caches sit under `flow_split/latents_<tag>/<split>/`. Reach all of
  it through `RunConfig.DATASETS["IIoTset-Ferrag"]` rather than hardcoding paths.
  `data_artefacts/deprecated_*` is the pre-merge tree — nothing reads it any more.
- Datasets are Polars DataFrames (parquet under `data_artefacts/`, gitignored) with the columns
  `proto_hierarchy`, `flow_key`, `timestamp_s`, `timestamp_us`, `data` (raw bytes), `mask`
  (which bytes to redact for MLM/PII) and `header_len`. There is **no** `AttackLabel` or
  `FlowID` column: attack labelling is by file name (`attacks/<Class>.parquet`) and the flow
  identifier is the `flow_key` string. `TrainingDatasetHandler` / `ValidationDatasetHandler` in
  `DataUtils.py` require those two missing columns and are therefore dead against current
  artefacts — use `PreTrainingDatasetHandler`.
- `data_artefacts/`, model checkpoints (`*.pth`, `*.ckpt`), and logs are gitignored — training
  scripts write into per-run subdirectories under `RawByteTrafficModelling/PreTraining/TrainingOutputs/`
  or `.../AnomalyDetection/Outputs/`. Scripts on `RunConfig.setup_run()` create their own
  output dir; the older ones do not, and their `logging.FileHandler` raises if it is missing.

# Artefacts

This file describes everything a stage writes and the next stage reads. Stage docs link
here and do not describe these formats again. None of these files are tracked in git.

## Directory layout

```
data/<capture>/...                                   source pcaps (outside the pipeline)
data_artefacts/
├── merged_extractor/                                DATA_ROOT in RunConfig.py
│   ├── logs/<capture>.log, summary.tsv              extractor stdout per capture
│   └── <capture>/                                   one DatasetPaths entry
│       ├── <source>.parquet                         normal capture (stage 1)
│       ├── attacks/<AttackClass>.parquet            one file per attack capture (stage 1)
│       └── flow_split/                              stage 2
│           ├── <role>.parquet                       one per split role
│           ├── split_report.json
│           └── latents_<CACHE_TAG>/<split>/         stage 4
│               ├── meta.json
│               ├── flow_offsets.parquet
│               └── shard_XXXX.npy
└── deprecated_*/                                    pre-merge extractor output, read by nothing

RawByteTrafficModelling/
├── PreTraining/TrainingOutputs/<RUN_NAME>/          stages 3 and 5
│   ├── <Prefix>_<RUN_NAME>_{untrained,latest,E<n>,best}.ckpt
│   ├── metrics.csv, curves.png, <Script>.log
└── AnomalyDetection/Outputs/
    ├── Embeddings/SequenceEmbeddings_<RUN_NAME>/    stage 6
    └── AD/<RUN_NAME>_eval-<split>/                  stage 7
```

## Dataset registry

All paths under `data_artefacts/` are built by `PreTraining/RunConfig.py::DatasetPaths`,
and each capture has one entry in `RunConfig.DATASETS`. Scripts select a capture with
`DATASET = DATASETS["<name>"]` and never hardcode a path.

| Field / method | Meaning |
|---|---|
| `name` | directory under `DATA_ROOT` |
| `source` | file name of the attack-free capture that was split |
| `splits` | every split name present in `flow_split/`. `split(name)` raises on any other name |
| `has_attacks` | whether `attacks/` exists. `attacks` is `None` when it does not |
| `train` / `test` / `val` | shorthand for `split("train")` and so on |
| `latent_cache(tag, split)` | `flow_split/latents_<tag>/<split>` |

The current entries have different split layouts; see
[02 § Current state](../components/02-temporal-split.md#current-state).

## Packet parquet (stage 1 → stage 2)

Written by `feature_extraction/src/main.rs::ParquetSink`, with one row per packet. The
split parquets (`flow_split/<role>.parquet`) and the attack parquets use the same schema.

| Column | Type | Content |
|---|---|---|
| `proto_hierarchy` | str | e.g. `Ethernet->IPv4->TCP->MQTT`, as deep as the parser got |
| `flow_key` | str | canonical bidirectional key `ip_a:port_a -> ip_b:port_b (PROTO)` |
| `timestamp_s`, `timestamp_us` | i64 | capture timestamp |
| `data` | binary | the full frame, starting at the Ethernet header |
| `mask` | binary | one byte per byte of `data`; `1` = endpoint-identifying byte |
| `header_len` | u32 | parsed header bytes (not used by the Python side at present) |

There is no label column. An attack's class is the file name of its parquet, and a
normal packet is any packet from the normal capture. There is no `FlowID` column
either: the flow identifier is the `flow_key` string. `ModelComponents/FlowID.py` and
`DataUtils.py::{Training,Validation}DatasetHandler` expect those columns, so they do not
work on current artefacts.

The format of `flow_key` is defined in `feature_extraction/src/flow_tracker.rs::FlowKey`
(`normalize`, `Display`) and parsed on the Python side by
`data_tools/SplitFlowsDF.py::FLOW_KEY_RE` / `DataUtils.py::parse_flow_key`. Ports are `0`
for ARP and ICMP.

## Split report

`split_report.json` is written by `SplitFlowsDF.split_flows`. It records the requested and
achieved ratios, the first and last timestamp and time span of each split, and
`flow_keys_crossing_a_boundary`. Check it after every split.

## Packet latent cache (stage 4 → stages 5, 6)

Written by `PreTraining/CachePacketLatents.py` and read by `DataUtils.py::load_latent_cache`.

| File | Content |
|---|---|
| `shard_XXXX.npy` | float16 `(rows, D)` packet latents, `SHARD_ROWS` per shard, in **cache row order** |
| `flow_offsets.parquet` | `flow_key`, `start`, `length`: each flow is the contiguous slice `latents[start:start+length]` |
| `meta.json` | `split_file`, `packet_ae_ckpt`, `packet_ae_sha256`, `latent_dim`, `num_rows`, `num_flows`, `shard_rows`, `num_shards`, `dtype` |

Cache row order is the order of `PreTrainingDatasetHandler.build_flow_index()`: flows
sorted by `flow_key`, and packets within a flow sorted by timestamp. It is **not** the
row order of the parquet. Any code that needs to link a cache row back to its bytes has
to rebuild that index; see
[data-representation § Flow index](data-representation.md#flow-index).

## Checkpoints (stages 3, 5 → later stages)

Written by `ModelDefinitions.py::save_checkpoint` as a `torch.save` dict with the keys
`epoch`, `model_state_dict`, `optimizer_state_dict`, `loss`, `config` (the
`dataclasses.asdict` of the params) and the extras from the training script
(`scheduler_state_dict`, `best_*`, `global_step`, eval metrics). Load a checkpoint only
through its matching `load_*` function; see
[model-library § Checkpoints](model-library.md#params-and-checkpoints).

| Suffix | Written when |
|---|---|
| `_untrained` | at the start of a fresh (non-resumed) run |
| `_latest` | mid-epoch, on the eval interval (packet level only). Stamped `epoch-1` so that resuming replays the unfinished epoch |
| `_E<n>` | end of epoch *n* |
| `_best` | end of an epoch that improves the selection metric on `test` |

Downstream stages always load `_best`. A sequence-AE checkpoint contains the frozen
packet encoder's weights under `encoder.packet_encoder.*`, so stage 6 does not need the
packet-AE checkpoint.

## Embedding directory (stage 6 → stage 7)

Written by `AnomalyDetection/SequenceEmbeddingAD.py` and read by
`AnomalyDetection/EmbeddingAD/data.py::load_embedding_dir`.

| File | Content |
|---|---|
| `<label>.npy` | float32 `(windows, D_seq)` flow vectors `z`. The label is a split name or an attack class |
| `metadata.parquet` | `set` (split name, or `attack`), `label`, `flow_key`, `seq_len`. Row *i* describes row *i* of the `.npy` files stacked in `sorted(glob("*.npy"))` order |

`load_embedding_dir` checks that alignment run by run and fails if it does not hold.

## AD outputs (stage 7)

`scores.parquet` (the metadata columns plus one `score_<detector>` column per detector),
`metrics.parquet` (long format: `detector`, `attack`, `bin`, `metric`, `value`), PNG
figures, `projection.html`, and `report.html`, which combines all of them.

## Lineage and provenance checks

A split, its caches, the checkpoints trained on them and the embeddings exported from
those checkpoints form one lineage, and they are only valid together. These checks
enforce that:

| Check | Where | Catches |
|---|---|---|
| `packet_ae_sha256` vs. the checkpoint on disk | `DataUtils.py::load_latent_cache` (via `checkpoint_fingerprint`) | a cache built from different packet-AE weights |
| `flow_key` order reproduces `flow_offsets` | `CheckLatentCache.py`, `SeqByteEval.py`, `SequenceEmbeddingAD.py` | a parquet that no longer matches its cache (e.g. after re-extraction) |
| `meta["split_file"] == VAL_SPLIT_FILE` | `SequenceLevelAutoEncoder.py`, `SeqByteEval.py` | byte eval that reads a different parquet from the one the cache was built on |
| cached vs. live-encoder latents | `CheckLatentCache.py`, `SeqByteEval.py`, `SequenceEmbeddingAD.py` (`CROSS_CHECK_*`) | a row-order mismatch that raises no error and silently corrupts the numbers |

Re-extracting a capture changes its `flow_key` strings, so everything downstream of the
parquet has to be regenerated.

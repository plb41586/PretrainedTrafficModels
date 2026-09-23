# Pipeline: from PCAP to anomaly detector

Start here. This file gives the order of the stages and says where each one is
documented. It does not describe how any stage works: that is in the linked component
docs, and anything shared between stages is in `reference/`.

```
 pcap ──(1)──▶ packet parquet ──(2)──▶ flow_split/<role>.parquet
                                              │
                                              ├──(3)──▶ packet-AE checkpoint ─────────┐
                                              │                                       │
                                              └──(4)──▶ latent cache (train, test) ◀──┘
                                                              │
                                                              ├──(5)──▶ sequence-AE checkpoint
                                                              │          (stage 1, optional stage 2)
                                                              ▼                 │
                          splits without a cache + attacks ──(6)──▶ flow embeddings (.npy + metadata)
                                                                                │
                                                                                └──(7)──▶ detectors, metrics, report.html
```

| # | Stage | Entry point | Consumes | Produces | Doc |
|---|---|---|---|---|---|
| 1 | Feature extraction | `feature_extraction/` (Rust CLI) | `.pcap` | packet parquet | [01](components/01-feature-extraction.md) |
| 2 | Temporal split | `data_tools/SplitFlowsDF.py` | normal packet parquet | `flow_split/*.parquet`, `split_report.json` | [02](components/02-temporal-split.md) |
| 3 | Packet-level pretraining | `PreTraining/PacketLevelAutoEncoder.py` (`PacketLevelMLM.py` is the alternative) | `train`, `test` splits | packet-AE checkpoint | [03](components/03-packet-pretraining.md) |
| 4 | Packet latent cache | `PreTraining/CachePacketLatents.py`, then `CheckLatentCache.py` | splits + packet-AE checkpoint | `flow_split/latents_<tag>/<split>/` | [04](components/04-packet-latent-cache.md) |
| 5 | Sequence-level pretraining | `PreTraining/SequenceLevelAutoEncoder.py`, optionally `SequenceAEStructureFinetune.py` | latent caches + packet-AE checkpoint | sequence-AE checkpoint | [05](components/05-sequence-pretraining.md) |
| 6 | Flow embedding export | `AnomalyDetection/SequenceEmbeddingAD.py` | sequence-AE checkpoint, caches, uncached splits, attack parquets | `Outputs/Embeddings/SequenceEmbeddings_<run>/` | [06](components/06-flow-embedding-export.md) |
| 7 | Anomaly detection | `AnomalyDetection/EmbeddingADSuite.py` | embedding directory | `Outputs/AD/<run>_eval-<split>/` | [07](components/07-anomaly-detection.md) |

Python paths in the table are relative to `RawByteTrafficModelling/`.

## Shared reference

| Topic | Doc |
|---|---|
| Every file a stage reads or writes: directory layout, schemas, naming, provenance checks | [reference/artefacts.md](reference/artefacts.md) |
| How bytes become model input: endpoint masking, token IDs, CLS placement, flow windows | [reference/data-representation.md](reference/data-representation.md) |
| `ModelComponents/`: backbones, params dataclasses, checkpoints, model classes | [reference/model-library.md](reference/model-library.md) |
| Running scripts (devcontainer), script anatomy, `RunConfig.py`, split-role naming | [reference/conventions.md](reference/conventions.md) |

## Adding a dataset

This is the full list of places to touch when a new capture goes through the pipeline.
Each script is configured through constants at the top of the file (see
[conventions](reference/conventions.md#script-anatomy)). The `Known gaps` sections of the
component docs list what does not work yet for a capture that differs from IIoTset-Ferrag.

1. **Extract** every pcap (the normal capture and each attack capture) to parquet
   → [01](components/01-feature-extraction.md). Check the `parse errors` count in the log.
2. **Register** the capture: add a `DatasetPaths` entry to `DATASETS` in
   `PreTraining/RunConfig.py`, with its `source`, `splits` and `has_attacks`
   → [artefacts](reference/artefacts.md#dataset-registry).
3. **Split** the normal capture: set `DATA_FILE`, `OUTPUT_DIR` and `SPLITS` in the
   `__main__` block of `SplitFlowsDF.py`. Try `DRY_RUN = True` first
   → [02](components/02-temporal-split.md).
4. **Packet AE**: set `RUN_NAME` and `DATASET`, run with `SMOKE = True` first, then do the real run
   → [03](components/03-packet-pretraining.md).
5. **Cache**: in `CachePacketLatents.py` set `DATASET`, `PACKET_AE_CKPT` and a new `CACHE_TAG`.
   Then set the same three in `CheckLatentCache.py` and run it
   → [04](components/04-packet-latent-cache.md).
6. **Sequence AE**: set `RUN_NAME`, `DATASET`, `CACHE_TAG` and `PACKET_AE_CKPT`, and check
   whether `Epochs` still gives a sensible step budget for the new flow count
   → [05](components/05-sequence-pretraining.md).
7. *(optional)* **Structure fine-tune**: profile the groups with `InspectFlowGroups.py`
   first, because the choice of grouping key depends on the capture
   → [05](components/05-sequence-pretraining.md#stage-2-structure-fine-tune).
8. **Export** embeddings: set `RUN_NAME`, `DATASET`, `CACHE_TAG`, `TOKEN_SETS`
   → [06](components/06-flow-embedding-export.md).
9. **Detect**: set `RUN_NAME` and the three role labels, then do a `SMOKE` run before the real
   run, because the real run spends the evaluation split
   → [07](components/07-anomaly-detection.md).

Keep one artefact lineage per capture. A re-extracted parquet invalidates its split and
every cache and checkpoint built downstream of it
([artefacts](reference/artefacts.md#lineage-and-provenance-checks)).

## Reusing the pipeline for other downstream tasks

Anomaly detection is one consumer. Each stage's output is a self-contained interface, and
another task can branch off at whichever one fits:

| Branch off at | You get | Suitable for | Reuse notes |
|---|---|---|---|
| Packet parquet (1) | raw frames + endpoint mask + `flow_key` + timestamps | any byte-level or flow-level model | [01 § Reuse](components/01-feature-extraction.md#reuse) |
| Split (2) | disjoint time intervals | any model that needs a leakage-free temporal hold-out | [02 § Reuse](components/02-temporal-split.md#reuse) |
| Packet encoder (3) | `Packet_Encoder`: 1520 tokens → one D-dim vector | packet classification (`Packet_Classifier`), packet-level AD, protocol ID | [03 § Reuse](components/03-packet-pretraining.md#reuse) |
| Latent cache (4) | per-packet vectors grouped by flow | any sequence model over packets, trained without the packet forward pass | [04 § Reuse](components/04-packet-latent-cache.md#reuse) |
| Sequence encoder (5) | `Sequence_Encoder`: ≤64 packets → one flow vector `z` | flow classification, clustering, device fingerprinting, fine-tuning | [05 § Reuse](components/05-sequence-pretraining.md#reuse) |
| Embedding export (6) | `<label>.npy` + `metadata.parquet` | anything that consumes fixed vectors (sklearn, UMAP, retrieval) | [06 § Reuse](components/06-flow-embedding-export.md#reuse) |

## How these docs are organised

- **Each fact is written down once.** A stage doc covers what the stage does and why.
  File formats are in `reference/artefacts.md`, classes in `reference/model-library.md`,
  and how to run things in `reference/conventions.md`. Other docs link to these instead
  of repeating them.
- **Code is referenced by path and symbol** (for example `DataUtils.py::load_latent_cache`),
  never by line number, so a reference stays correct when code moves within a file.
- **Hyperparameter values are not copied here.** They are constants in the scripts and
  change from run to run. A doc names the constant, and the script's own comments explain
  the value it currently has.
- **Results are not recorded here.** Numbers belong with their run in `TrainingOutputs/`
  or `Outputs/`, and in the commit that produced them.
- **`docs/` is the only description of the project.** `CLAUDE.md` contains no
  description. It holds the coding agent's working rules and the list of invariants,
  each of which links back to the doc that explains it.
- **Update the docs in the same change as the code.** Renaming a symbol, adding a stage,
  or changing an artefact format means updating the owning doc too, since a stale doc is
  worse than none. `TODO.md` holds open work items and is not documentation.

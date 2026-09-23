# 04 · Packet latent cache (split + packet encoder → cached latents)

**What it does:** runs the frozen packet encoder once over every packet of a split and
stores the latents grouped by flow. Sequence-level training then fetches batches from
this cache by index lookup, without running 64 packet-encoder forwards per window at
every step.

**Entry points** (in `RawByteTrafficModelling/PreTraining/`):

```
python -m RawByteTrafficModelling.PreTraining.CachePacketLatents   # builds every split in SPLITS
python -m RawByteTrafficModelling.PreTraining.CheckLatentCache     # read-only verification of one split
```

**Consumes:** `DATASET.split(<s>)` for each split in `SPLITS`, and the packet-AE `_best`
checkpoint (`PACKET_AE_CKPT`).
**Produces:** `flow_split/latents_<CACHE_TAG>/<split>/`
([artefacts § latent cache](../reference/artefacts.md#packet-latent-cache-stage-4--stages-5-6)).

## Where the logic lives

| Concern | File · symbol |
|---|---|
| Build | `CachePacketLatents.py::cache_split`, `encode_rows` |
| Row order | `DataUtils.py::PreTrainingDatasetHandler.build_flow_index` ([data-representation § Flow index](../reference/data-representation.md#flow-index)) |
| Load and check provenance | `DataUtils.py::load_latent_cache`, `checkpoint_fingerprint` |
| Batching from the cache | `DataUtils.py::CachedLatentSequenceHandler` ([data-representation § Flow → windows](../reference/data-representation.md#flow--windows)) |
| Verification | `CheckLatentCache.py`: live-vs-cached equivalence, handler shapes, zero padding, flow-length histogram |

## How it works

1. Build the flow index. The concatenated row indices give the cache row order, and the
   cumulative lengths give each flow's `start`.
2. Encode in fp32 (`ENCODE_DTYPE`) and store as float16, in shards of `SHARD_ROWS`. A
   shard that already exists with the right shape is skipped, so an interrupted build
   resumes where it stopped.
3. Write `meta.json` last, including the sha256 of the checkpoint. `load_latent_cache`
   refuses a cache whose checkpoint hash does not match.

Only `train` and `test` are cached, because those are the splits the sequence level
trains and selects on. The AD splits and the attack captures go through the token path
in stage 6 instead.

## Configuration

`DATASET`, `PACKET_AE_CKPT`, `CACHE_TAG` (**change it whenever the checkpoint changes**, so
that two caches from different checkpoints never share a directory), `SPLITS`,
`DEVICE_INDEX`. `CheckLatentCache.py` must use the same `DATASET`, `CACHE_TAG` and
`PACKET_AE_CKPT`, plus the `SPLIT` to check.

## Reuse

Any sequence model over packets can train on the cache without running the packet
encoder: `load_latent_cache` → `CachedLatentSequenceHandler` → `(B, P, D)` latents +
`seq_lens`. `flow_offsets.flow_key` links each window to its flow, and from there to
endpoint groups (`DataUtils.flow_group_ids`) or any other flow-level label.

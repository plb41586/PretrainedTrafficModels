# 06 · Flow embedding export (sequence AE → embedding directory)

**What it does:** runs a trained sequence encoder over deterministic windows of the
normal splits and attack captures named in `DATA_ROLES`, and writes one flow vector `z`
per window.
Consumers of the export (stage 7, UMAP plots) need no torch code.

**Entry point:** `RawByteTrafficModelling/AnomalyDetection/SequenceEmbeddingAD.py`.

**Consumes:** a sequence-AE checkpoint (`RUN_NAME`, `CKPT_SUFFIX`: `_best` by default), the latent caches of the
`cached` splits, the split parquets of the `token` splits, and the `attack` captures
(`DATASET.attack(name)`).
**Produces:** `Outputs/Embeddings/SequenceEmbeddings_<RUN_NAME>[_<CKPT_SUFFIX>]/` (the suffix only when it is not `best`), containing `<label>.npy`
and `metadata.parquet` ([artefacts § Embedding directory](../reference/artefacts.md#embedding-directory-stage-6--stage-7)).

## Where the logic lives

| Concern | Symbol in `SequenceEmbeddingAD.py` |
|---|---|
| What is read, as what; one log line per read | `DATA_ROLES`, `roles` (`RunConfig.DataRoles`, [conventions § Data roles](../reference/conventions.md#data-roles)) |
| Windowing (same rule as `CachedLatentSequenceHandler.enumerate_windows`) | `enumerate_windows`, `pick_windows` (evenly spaced subset up to `MAX_WINDOWS_PER_SET`) |
| Cache path | `embed_cached` |
| Token path: encode real packets from bytes, leave padding as zeros | `embed_tokens`, `token_latent_batch`, `flow_offsets_from_index` |
| Check that both paths agree | the `CROSS_CHECK_*` block |
| Metadata ordering | the final block, sorted by `f"{label}.npy"` |

## How it works

There are two ways to get packet latents, and both feed the same `Sequence_Encoder`:

- **Cached** (role `cached`: `train` and `test`, the splits stage 4 caches): the latents
  are read from the stage-4 cache, with no packet forward pass.
- **Token** (role `token`: `ad_fit`, `ad_calib`, `val`, `late`, plus every `attack`
  capture): the bytes go through the packet encoder inside the sequence-AE checkpoint.
  Windows are formed from `build_flow_index()`, so they come out identical to the
  cached path.

Before anything is exported, the cross-check encodes the first `CROSS_CHECK_WINDOWS`
windows of the `cross_check` split (`test`) both ways, and asserts that they agree to within
`CROSS_CHECK_TOL`. A row-order mismatch would not raise an error; it would silently
produce wrong embeddings, and this check is what catches it. `cross_check` is the only
role allowed to share a split with another role (`cached`). A split listed under both
`cached` and `token` raises an error, because it would be exported twice under one label.

A capture with no attacks (`has_attacks=False`) resolves `ALL_ATTACKS` to nothing, so the
attack loop is skipped.

Attack class = file name. The `set` column is `"attack"` for all attack files, and the
split name for normal sets.

## Configuration

`RUN_NAME` (the run to export), `CKPT_SUFFIX` (which of its checkpoints; `best` by default), `DATASET`, `CACHE_TAG`, `DATA_ROLES`,
`MAX_WINDOWS_PER_SET`, `SMOKE` (two attack files and small caps, written to a separate
directory).

## Reuse

The export directory is the stage's interface: numpy arrays plus a metadata table that
has `flow_key` and `seq_len` for every row. Any vector-based consumer can read it: a
classifier over attack classes, clustering, retrieval, or `embedding_viz.py`. To export
a different model, point `RUN_NAME` at any checkpoint that `load_SeqAE_checkpoint` can
load.

## Known gaps

- `CLS_Placement="EOS"` is repeated inline here instead of coming from
  `RunConfig.make_id_encoder()`.

# 06 · Flow embedding export (sequence AE → embedding directory)

**What it does:** runs a trained sequence encoder over deterministic windows of every
normal split and every attack capture, and writes one flow vector `z` per window.
Consumers of the export (stage 7, UMAP plots) need no torch code.

**Entry point:** `RawByteTrafficModelling/AnomalyDetection/SequenceEmbeddingAD.py`.

**Consumes:** the sequence-AE `_best` checkpoint (`RUN_NAME`), the latent caches for the
`CACHED_SETS`, the split parquets for the `TOKEN_SETS`, and `DATASET.attacks/*.parquet`.
**Produces:** `Outputs/Embeddings/SequenceEmbeddings_<RUN_NAME>/`, containing `<label>.npy`
and `metadata.parquet` ([artefacts § Embedding directory](../reference/artefacts.md#embedding-directory-stage-6--stage-7)).

## Where the logic lives

| Concern | Symbol in `SequenceEmbeddingAD.py` |
|---|---|
| Windowing (same rule as `CachedLatentSequenceHandler.enumerate_windows`) | `enumerate_windows`, `pick_windows` (evenly spaced subset up to `MAX_WINDOWS_PER_SET`) |
| Cache path | `embed_cached` |
| Token path: encode real packets from bytes, leave padding as zeros | `embed_tokens`, `token_latent_batch`, `flow_offsets_from_index` |
| Check that both paths agree | the `CROSS_CHECK_*` block |
| Metadata ordering | the final block, sorted by `f"{label}.npy"` |

## How it works

There are two ways to get packet latents, and both feed the same `Sequence_Encoder`:

- **Cached** (`CACHED_SETS`, normally `train` and `test`): the latents are read from the
  stage-4 cache, with no packet forward pass.
- **Token** (`TOKEN_SETS` and every attack file): the bytes go through the packet encoder
  inside the sequence-AE checkpoint. Windows are formed from `build_flow_index()`, so
  they come out identical to the cached path.

Before any attack set is written, the cross-check encodes the first `CROSS_CHECK_WINDOWS`
windows of `test` both ways and asserts that they agree to within `CROSS_CHECK_TOL`. A
row-order mismatch would not raise an error; it would silently produce wrong
embeddings, and this check is what catches it.

Attack class = file name. The `set` column is `"attack"` for all attack files, and the
split name for normal sets.

## Configuration

`RUN_NAME` (the checkpoint to export), `DATASET`, `CACHE_TAG`, `CACHED_SETS`, `TOKEN_SETS`,
`MAX_WINDOWS_PER_SET`, `SMOKE` (two attack files and small caps, written to a separate
directory).

## Reuse

The export directory is the stage's interface: numpy arrays plus a metadata table that
has `flow_key` and `seq_len` for every row. Any vector-based consumer can read it: a
classifier over attack classes, clustering, retrieval, or `embedding_viz.py`. To export
a different model, point `RUN_NAME` at any checkpoint that `load_SeqAE_checkpoint` can
load.

## Known gaps

These apply to a capture that differs from IIoTset:

- `ATTACK_DIR = DATASET.attacks` is passed straight to `os.listdir`. For a capture with
  `has_attacks=False` (currently `CICAPT-IIoT`) it is `None` and the export fails once
  the normal sets are done.
- `TOKEN_SETS` lists only `val`. In the six-role layout, `ad_fit`, `ad_calib` and `late`
  have no cache, so they also have to be listed in `TOKEN_SETS` before stage 7 can use
  them.
- `CLS_Placement="EOS"` is repeated inline here instead of coming from
  `RunConfig.make_id_encoder()`.

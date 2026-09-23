# 07 · Anomaly detection (flow embeddings → detectors and report)

**What it does:** fits one-class detectors on normal flow embeddings only, sets
thresholds from normal scores without looking at a label, and then scores held-out
normals against the attack captures.

**Entry point:** `RawByteTrafficModelling/AnomalyDetection/EmbeddingADSuite.py`.
The reusable parts are in the `AnomalyDetection/EmbeddingAD/` package.

**Consumes:** an embedding directory from stage 6 (`EMBEDDING_DIR`).
**Produces:** `Outputs/AD/<RUN_NAME>_eval-<EVAL_NEG_LABEL>/`
([artefacts § AD outputs](../reference/artefacts.md#ad-outputs-stage-7)).

## Where the logic lives

| Concern | File · symbol (under `AnomalyDetection/`) |
|---|---|
| Driver: role assignment, fit/score, metrics, figures | `EmbeddingADSuite.py` · `select_detectors`, `fit_and_score` |
| Load and check alignment | `EmbeddingAD/data.py` · `load_embedding_dir` |
| Role masks (disjoint, non-empty; halves one set if it has two roles) | `EmbeddingAD/data.py` · `split_roles` |
| Preprocessing, fitted on the fit set only | `EmbeddingAD/data.py` · `Preprocessor` (`l2_standardize` / `standardize` / `raw`, optional PCA) |
| Detector interface and registry | `EmbeddingAD/detectors.py` · `Detector`, `DETECTORS`, `SLOW_DETECTORS` |
| Thresholds and metrics | `EmbeddingAD/evaluation.py` · `calibrate`, `evaluate`, `bin_masks`, `agreement_matrix`, `pivot`, `pivot_bins` |
| Figures and HTML report | `EmbeddingAD/plots.py` · `plot_*`, `write_report`, `table_html` |

## How it works

1. **Assign roles.** Each exported label gets one role: `FIT_LABEL` → `fit`,
   `CALIB_LABEL` → `calib`, `EVAL_NEG_LABEL` → `eval_neg`, and `set == "attack"` →
   `eval_pos`. Rows with no role are dropped before anything is scored or plotted, so
   an unused held-out split cannot show up in the figures either.
2. **Preprocess** with parameters fitted on `fit` alone.
3. **Fit** every detector in `DETECTORS` on `fit`, and score every row. Higher scores mean
   more anomalous.
4. **Calibrate** thresholds as quantiles (`QUANTILES`) of the `calib` scores. This targets
   a false-positive rate without using a label. The report then compares the achieved
   FPR on `eval_neg` against the target, and a large gap there means the normal
   distribution drifted.
5. **Evaluate:** labels are used for the first time here. AUROC, AUPRC, partial AUROC
   (FPR ≤ 1%) and TPR at each calibrated threshold are computed pooled, per attack class,
   and per `seq_len` bin.
6. **Report:** heatmaps, ROC/PR curves, score distributions, score vs. length,
   calibration, detector agreement, and a projection of `FOCUS_DETECTOR` where UMAP is
   fitted on normals only. Everything is combined into `report.html`.

The reported operating points are TPR at a calibrated FPR, not accuracy or F1. The
pooled prevalence depends only on how many windows each attack file happens to contain,
which would make F1 meaningless.

### Length confound

The attack exports consist mostly of very short windows, while normal windows are
usually full length, so a pooled AUROC largely measures flow length. Two things control
for this:

- `seq_len_only` is registered as a detector. A learned detector only shows something
  about the embedding when it beats this baseline.
- Every table is repeated per `SEQ_LEN_BINS` bin, with length-matched normals and a
  bin-local threshold. `MIN_BIN_N` sets the minimum count below which a bin is reported
  as NaN.

Read the stratified table before the pooled one.

### Detector selection

`FOCUS_DETECTOR` is fixed in the config, not chosen from the metrics. Choosing the best
detector from the same labelled rows it is then reported on would bias the result.

## Configuration

`RUN_NAME`/`EMBEDDING_DIR`, the three `*_LABEL` roles, `FOCUS_DETECTOR`, `PREPROCESS`,
`PCA_COMPONENTS`, `QUANTILES`, `SEQ_LEN_BINS`, `ENABLE_SLOW`, `REFIT` (`False` reuses
`scores.parquet` and redoes only the metrics and figures), `SMOKE`.

Every run that reads the evaluation split uses it up, so check the wiring with `SMOKE`
first.

## Known gaps

- **The suite has not been moved to the six-role layout.** It still uses `train` / `test` /
  `val` for fit / calib / eval_neg. The six-role layout
  ([02 § Split roles](02-temporal-split.md#split-roles)) is designed for `ad_fit` /
  `ad_calib` / `val`, which needs those splits exported first
  ([06 § Known gaps](06-flow-embedding-export.md#known-gaps)).

## Extending

- **New detector:** a class with `fit(X, meta)` / `score(X, meta) → (N,)`, plus one
  entry in `DETECTORS`. `meta` is passed in, so a detector can also use `seq_len` or
  `flow_key`.
- **Different embeddings:** the suite needs only an embedding directory, so packet-level
  or other flow embeddings work if they are exported in the same format.

## Other scripts

| Script | Status |
|---|---|
| `embedding_viz.py` | UMAP fitted on `test` normals alone, with attacks transformed into it. Three views (overview, one panel per attack, coloured by `seq_len`). Read the length view before trusting the overview |
| `Embedding_Viz.ipynb` | exploratory only. It fits the scaler and UMAP on normals and attacks together, so its separation is not evidence |
| `AutoEncoderAD.py` | legacy packet-level AD by reconstruction loss |
| `EncoderEmbeddingAD.py` | legacy export of packet-level embeddings |
| `VisualizeDistribution.py` | generic interactive histogram viewer for a directory of `.npy` files |

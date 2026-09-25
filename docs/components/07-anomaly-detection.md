# 07 · Anomaly detection (flow embeddings → detectors and report)

**What it does:** fits one-class detectors on normal flow embeddings only, sets
thresholds from normal scores without looking at a label, and then scores held-out
normals against the attack captures.

**Entry point:** `RawByteTrafficModelling/AnomalyDetection/EmbeddingADSuite.py`.
The reusable parts are in the `AnomalyDetection/EmbeddingAD/` package.

**Consumes:** the sets named in `DATA_ROLES`, from an embedding directory written by stage 6 (`EMBEDDING_DIR`).
**Produces:** `Outputs/AD/<RUN_NAME>_eval-<eval_neg sections>/`
([artefacts § AD outputs](../reference/artefacts.md#ad-outputs-stage-7)).

## Where the logic lives

| Concern | File · symbol (under `AnomalyDetection/`) |
|---|---|
| Driver: role map, fit/score, metrics, drift table, figures | `EmbeddingADSuite.py` · `DATA_ROLES`, `select_detectors`, `fit_and_score` |
| Role map validation and access log | `PreTraining/RunConfig.py` · `DataRoles` ([conventions § Data roles](../reference/conventions.md#data-roles)) |
| Which sets the directory holds | `EmbeddingAD/data.py` · `embedding_sections` |
| Load only the mapped sets, and check alignment | `EmbeddingAD/data.py` · `load_embedding_dir` |
| Role masks (disjoint; required roles non-empty) | `EmbeddingAD/data.py` · `split_roles` |
| Preprocessing, fitted on the fit set only | `EmbeddingAD/data.py` · `Preprocessor` (`l2_standardize` / `standardize` / `raw`, optional PCA) |
| Detector interface and registry | `EmbeddingAD/detectors.py` · `Detector`, `DETECTORS`, `SLOW_DETECTORS` |
| Thresholds and metrics | `EmbeddingAD/evaluation.py` · `calibrate`, `evaluate`, `bin_masks`, `agreement_matrix`, `pivot`, `pivot_bins` |
| Figures and HTML report | `EmbeddingAD/plots.py` · `plot_*`, `write_report`, `table_html` |

## How it works

1. **Assign roles.** `DATA_ROLES` maps each role to exported sets. By default `fit` →
   `ad_fit`, `calib` → `ad_calib`, `eval_neg` → `val`, `eval_pos` → every attack set, and
   `drift` → `late`. No set may serve two roles. Only the mapped `.npy` files are read,
   and each read is logged. An unmapped split is never loaded, so it cannot appear in
   the scores, the figures or `scores.parquet`.
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
   If `drift` is mapped, the same metrics are computed a second time with `drift` as the
   negatives. The detectors and thresholds are unchanged. Both sets of rows go into
   `metrics.parquet`, told apart by the `negatives` column. A drift table compares
   pooled AUROC and realised FPR between `eval_neg` and `drift`, which shows how far
   the operating point moves one split later in time.
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

`RUN_NAME`/`EMBEDDING_DIR`, `DATA_ROLES`, `FOCUS_DETECTOR`, `PREPROCESS`,
`PCA_COMPONENTS`, `QUANTILES`, `SEQ_LEN_BINS`, `ENABLE_SLOW`, `REFIT` (`False` reuses
`scores.parquet` and redoes only the metrics and figures), `SMOKE`.

Every run that reads the evaluation split uses it up, so check the wiring with `SMOKE`
first.

## Known gaps

- **CICAPT-IIoT cannot be evaluated.** It has no attack captures, so `eval_pos` selects
  nothing and `split_roles` raises.

## Extending

- **New detector:** a class with `fit(X, meta)` / `score(X, meta) → (N,)`, plus one
  entry in `DETECTORS`. `meta` is passed in, so a detector can also use `seq_len` or
  `flow_key`.
- **Different embeddings:** the suite needs only an embedding directory, so packet-level
  or other flow embeddings work if they are exported in the same format.

## Other scripts

| Script | Status |
|---|---|
| `embedding_viz.py` | UMAP fitted on the `reference` split alone (`test`), with attacks transformed into it. It reads only its `DATA_ROLES`. Three views (overview, one panel per attack, coloured by `seq_len`). Read the length view before trusting the overview |
| `Embedding_Viz.ipynb` | exploratory only. It fits the scaler and UMAP on normals and attacks together, so its separation is not evidence |
| `VisualizeDistribution.py` | generic interactive histogram viewer for a directory of `.npy` files |

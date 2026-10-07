"""Anomaly detection over exported sequence embeddings, fitted on normals only.

Embedding_Viz.ipynb shows promising separation between normal and attack flows, but it
cannot be used as evidence: both the StandardScaler and the UMAP reducer there are fitted
on the pooled normals *and* attacks, so the projection was told where the attacks are.
This script fits every transform and every detector on the fit-role normals alone,
calibrates thresholds from held-out normal scores without looking at a label, and only
then reveals labels to compute the reported metrics against the evaluation normals + the
attack exports.

The roles, with the six-split layout they are designed for (DATA_ROLES below):

    fit       ad_fit      preprocessing + every detector
    calib     ad_calib    score quantiles -> thresholds (no labels)
    eval_neg  val         -> y=0, the reported false-positive rate
    eval_pos  attacks     -> y=1
    drift     late        -> y=0, scored with the same detectors and thresholds and
                             reported beside val: how far the FPR moves one split later

None of these splits trained the AEs or picked their checkpoint (that is train / test).
Every rerun spends `val` again. The splits are separated by *time*, not by flow identity
-- SplitFlowsDF cuts long-lived conversations at a timestamp, so the same flow_key can
appear either side of a boundary and `val` is a held-out period rather than a held-out
population.

Only the sets DATA_ROLES maps are read from disk, each read logged as a DATA ACCESS line.
An unmapped set is never loaded -- not merely excluded from the metrics, which would
still put it in the projection figure and in scores.parquet.

Two things to read the output with. First, `seq_len_only` is in the detector list on
purpose: most attack exports are dominated by single-packet windows (DDoS_UDP_Flood is
100% single-packet) while the normal splits have a median of 64, so a pooled AUROC largely
measures flow length. A learned detector has shown something about the embedding only
where it beats that row. Second, every table is repeated per seq_len bin with
length-matched normals, which is where that confound is controlled rather than flagged.

Run from the repo root:
    python -m RawByteTrafficModelling.AnomalyDetection.EmbeddingADSuite
"""
import logging
import os
import time

import numpy as np
import polars as pl

from RawByteTrafficModelling.AnomalyDetection.EmbeddingAD import evaluation as ev
from RawByteTrafficModelling.AnomalyDetection.EmbeddingAD import plots
from RawByteTrafficModelling.AnomalyDetection.EmbeddingAD.data import (
    Preprocessor,
    embedding_sections,
    load_embedding_dir,
    split_roles,
    subsample,
    take_rows,
)
from RawByteTrafficModelling.AnomalyDetection.EmbeddingAD.detectors import (
    DETECTORS,
    SLOW_DETECTORS,
)
from RawByteTrafficModelling.PreTraining.RunConfig import ALL_ATTACKS, DataRoles

### Config
RUN_NAME = "SeqAE_IIoTset_d128_Mamba_s512"
EMBEDDING_DIR = f"RawByteTrafficModelling/AnomalyDetection/Outputs/Embeddings/SequenceEmbeddings_{RUN_NAME}"

# Which exported sets play which role -- the only sets this run reads. Sections are the
# labels in EMBEDDING_DIR (split names, attack stems); a role may take several. Roles must
# not share a section. `drift` is optional: map it to () to leave `late` unread.
#
# Every rerun spends `val` again, so rerun only for a wiring failure, never to improve a
# number.
DATA_ROLES = {
    "fit":      ("ad_fit",),
    "calib":    ("ad_calib",),
    "eval_neg": ("val",),
    "eval_pos": (ALL_ATTACKS,),
    "drift":    (),            # ("late",) where the capture has one; IIoTset currently does not
}

# The evaluation split names the output directory, so runs evaluated against different
# negatives land side by side instead of overwriting each other.
output_dir = (f"RawByteTrafficModelling/AnomalyDetection/Outputs/AD/"
              f"{RUN_NAME}_eval-{'+'.join(DATA_ROLES['eval_neg'])}")

# The detector this run is about, pinned rather than ranked. Picking the winner from the
# metrics below would select a detector on the same labelled rows it is then reported
# from -- a selection bias on the held-out split.
FOCUS_DETECTOR = "mahalanobis"

PREPROCESS = "l2_standardize"  # l2_standardize | standardize | raw -- fitted on `fit` only
PCA_COMPONENTS = None          # e.g. 64 or 0.95; speeds up the distance-based detectors

QUANTILES = (0.95, 0.99, 0.999)          # of the calibration normals -> 5% / 1% / 0.1% FPR
HEADLINE_TPR = "tpr@q0.99"               # operating point shown in the per-attack heatmap
SEQ_LEN_BINS = ((1, 1, "1"), (2, 8, "2-8"), (9, 32, "9-32"), (33, 64, "33-64"))
MIN_BIN_N = 30                 # fewer rows than this on either side -> NaN, not a number

ENABLE_SLOW = False            # include SLOW_DETECTORS (ocsvm); adds several minutes
PROJECTION_N = 50_000          # rows kept for the UMAP/PCA projection (transform is slow)
SEED = 0
REFIT = True                # False: reuse scores.parquet and only redo metrics + figures

# Wiring test: caps every set, drops the expensive detectors and the projection, and writes
# to a separate directory so a smoke run never overwrites real results.
SMOKE = False
N_MAX_PER_SET = 0              # 0 = keep everything

if SMOKE:
    N_MAX_PER_SET = 2000
    PROJECTION_N = 4000
    output_dir = f"{output_dir}_smoke"
    SMOKE_DETECTORS = ("seq_len_only", "mahalanobis", "pca_recon", "knn_20")

os.makedirs(output_dir, exist_ok=True)   # logging.FileHandler below needs it to exist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{output_dir}/EmbeddingADSuite.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

roles = DataRoles(DATA_ROLES, embedding_sections(EMBEDDING_DIR), logger)
plots.set_role_sections(roles.roles)

META_COLS = ["set", "label", "flow_key", "seq_len"]


def select_detectors() -> dict:
    """The registry entries this run uses, in registry order."""
    names = list(DETECTORS)
    if SMOKE:
        names = [n for n in names if n in SMOKE_DETECTORS]
    elif not ENABLE_SLOW:
        names = [n for n in names if n not in SLOW_DETECTORS]
    return {n: DETECTORS[n] for n in names}


def fit_and_score(Z: np.ndarray, meta: pl.DataFrame,
                  masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fit each detector on the fit split and score every row."""
    Z_fit, meta_fit = Z[masks["fit"]], meta.filter(pl.Series(masks["fit"]))
    scores = {}
    for name, factory in select_detectors().items():
        t0 = time.time()
        det = factory()
        det.fit(Z_fit, meta_fit)
        s = det.score(Z, meta)
        if s.shape[0] != Z.shape[0]:
            raise ValueError(f"{name} returned {s.shape[0]} scores for {Z.shape[0]} rows")
        scores[name] = s
        logger.info(f"{name}: fitted on {Z_fit.shape[0]} rows and scored {Z.shape[0]} "
                    f"in {time.time() - t0:.1f}s")
    return scores


# --- Load -------------------------------------------------------------------
X, meta = load_embedding_dir(EMBEDDING_DIR, roles)
if N_MAX_PER_SET:
    idx = subsample(meta, N_MAX_PER_SET, SEED)
    X, meta = X[idx], take_rows(meta, idx)
    logger.info(f"Subsampled to {X.shape[0]} rows ({N_MAX_PER_SET} per set)")

masks = split_roles(meta, roles)
# load_embedding_dir reads only mapped sets, so every row has a role. An unmapped row here
# would reach the projection figure and scores.parquet -- a look at data the run was not
# meant to see.
assert np.logical_or.reduce(list(masks.values())).all(), "a loaded row plays no role"

scores_path = f"{output_dir}/scores.parquet"

# --- Fit + score ------------------------------------------------------------
if REFIT or not os.path.exists(scores_path):
    pre = Preprocessor(PREPROCESS, PCA_COMPONENTS, SEED).fit(X[masks["fit"]])
    Z = pre.transform(X)
    logger.info(f"Preprocessed with mode={PREPROCESS} pca={PCA_COMPONENTS} -> {Z.shape}")
    scores = fit_and_score(Z, meta, masks)
    (meta.select(META_COLS)
     .with_columns([pl.Series(f"score_{n}", s) for n, s in scores.items()])
     .write_parquet(scores_path))
    logger.info(f"Wrote {scores_path}")
else:
    cached = pl.read_parquet(scores_path)
    if cached.height != meta.height or (cached["label"].to_numpy() != meta["label"].to_numpy()).any():
        raise ValueError(f"{scores_path} has {cached.height} rows that do not line up with the "
                         f"{meta.height} loaded here -- rerun with REFIT = True")
    scores = {c.removeprefix("score_"): cached[c].to_numpy()
              for c in cached.columns if c.startswith("score_")}
    logger.info(f"Reusing cached scores for {list(scores)} from {scores_path}")

# --- Metrics ----------------------------------------------------------------
logger.info(f"Scoring against {roles.sections('eval_neg')} normals "
            f"(labels enter here for the first time)")
metrics = ev.evaluate(scores, meta, masks, QUANTILES, SEQ_LEN_BINS, MIN_BIN_N)
all_metrics = [metrics]
if "drift" in masks:
    logger.info(f"Scoring the drift role {roles.sections('drift')} with the same detectors "
                f"and thresholds")
    drift_metrics = ev.evaluate(scores, meta, masks, QUANTILES, SEQ_LEN_BINS, MIN_BIN_N,
                                neg_role="drift")
    all_metrics.append(drift_metrics)
# One file, told apart by the `negatives` column; the figures below all read `metrics`
# (eval_neg) only.
pl.concat(all_metrics).write_parquet(f"{output_dir}/metrics.parquet")

agree = ev.agreement_matrix(scores, masks["eval_neg"] | masks["eval_pos"])
logger.info(f"Focus detector (pinned, not picked from these metrics): {FOCUS_DETECTOR}")

summary = (metrics
           .filter((pl.col("attack") == ev.POOLED) & (pl.col("bin") == ev.ALL_BIN)
                   & (pl.col("metric").is_in(["auroc", "auprc", "pauroc_fpr1pct", HEADLINE_TPR])))
           .pivot(on="metric", index="detector", values="value")
           .join(metrics
                 .filter((pl.col("attack") == ev.CALIBRATION) & (pl.col("bin") == ev.ALL_BIN)
                         & (pl.col("metric") == "fpr_evalneg@q0.99"))
                 .select(["detector", pl.col("value").alias("fpr_evalneg@q0.99")]),
                 on="detector", how="left"))
logger.info("Pooled summary (all seq_len -- inflated by the length confound):\n"
            + str(summary))

stratified = ev.pivot_bins(metrics, "auroc", ev.POOLED)
logger.info("Pooled AUROC by seq_len bin (length-matched normals):\n" + str(stratified))

drift_table = None
if "drift" in masks:
    # Same detectors, same calib thresholds, negatives one split later. A drop in AUROC or
    # a rise in realised FPR from eval_neg to drift is how much the normal traffic moved.
    def _neg_view(m: pl.DataFrame, tag: str) -> pl.DataFrame:
        keep = ["auroc"] + [f"fpr_evalneg@q{q}" for q in QUANTILES]
        return (m.filter((pl.col("bin") == ev.ALL_BIN)
                         & (pl.col("attack").is_in([ev.POOLED, ev.CALIBRATION]))
                         & pl.col("metric").is_in(keep))
                .with_columns(pl.concat_str([
                    pl.col("metric").str.replace("fpr_evalneg", "fpr", literal=True),
                    pl.lit(f" [{tag}]")]).alias("metric"))
                .pivot(on="metric", index="detector", values="value"))
    drift_table = _neg_view(metrics, "+".join(roles.sections("eval_neg"))).join(
        _neg_view(drift_metrics, "+".join(roles.sections("drift"))), on="detector")
    logger.info("Drift: eval_neg vs drift negatives (pooled AUROC, realised FPR):\n"
                + str(drift_table))

# --- Figures ----------------------------------------------------------------
figs = [
    ("Per-attack performance", "png",
     plots.plot_attack_heatmaps(metrics, HEADLINE_TPR, f"{output_dir}/attack_heatmaps.png")),
    ("Length-stratified performance", "png",
     plots.plot_stratified(metrics, FOCUS_DETECTOR, f"{output_dir}/stratified_heatmaps.png")),
    ("ROC / PR", "png", plots.plot_roc_pr(scores, masks, f"{output_dir}/roc_pr.png")),
    ("Score distributions", "png",
     plots.plot_score_distributions(scores, masks, metrics, QUANTILES,
                                    f"{output_dir}/score_distributions.png")),
    ("Score vs flow length", "png",
     plots.plot_score_vs_seqlen(scores, meta, masks, SEQ_LEN_BINS,
                                f"{output_dir}/score_vs_seqlen.png")),
    ("Threshold calibration", "png",
     plots.plot_calibration(metrics, QUANTILES, f"{output_dir}/calibration.png")),
    ("Detector agreement", "png",
     plots.plot_agreement(agree, f"{output_dir}/agreement.png")),
]

if not SMOKE:
    png, go_fig = plots.plot_projection(
        X, meta, masks, scores[FOCUS_DETECTOR], FOCUS_DETECTOR,
        out_html=f"{output_dir}/projection.html", out_png=f"{output_dir}/projection.png",
        n_projection=PROJECTION_N, seed=SEED)
    figs += [("Normals-only projection", "png", png),
             ("Normals-only projection (interactive)", "plotly", go_fig)]

tables = [("Pooled summary (length-confounded -- read the stratified table below)", "html",
           plots.table_html(summary)),
          ("Pooled AUROC by seq_len bin (length-matched)", "html",
           plots.table_html(stratified))]
if drift_table is not None:
    tables.append(("Temporal drift: eval normals vs the later drift split", "html",
                   plots.table_html(drift_table)))
figs = tables + figs

report = plots.write_report(f"{output_dir}/report.html", figs)
logger.info(f"Wrote {report}")
logger.info(f"Done. {len(scores)} detectors, {meta.height} rows, outputs in {output_dir}")

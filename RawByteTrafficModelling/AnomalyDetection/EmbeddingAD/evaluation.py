"""Label-free threshold calibration and labelled scoring of the detector outputs.

Two rules the whole module is built around:

1. **Thresholds never see a label.** They are quantiles of the `test`-normal scores, which
   targets a false-positive rate by construction. Whether the target actually holds is then
   measured on the held-out `val` normals -- a large gap there means the normal manifold
   moved between splits and the detector is not deployable however good its AUROC looks.
2. **Length-matched comparison, not just pooled.** The attack exports are almost all
   single-packet windows and the normal splits almost none, so a pooled AUROC mostly
   measures flow length. Every table is therefore repeated per `seq_len` bin, with attack
   windows scored only against normal windows from the same bin and a bin-local threshold.

Reported operating points are TPR at a calibrated FPR rather than accuracy or F1: the
pooled prevalence here is an artefact of how many windows each attack file happens to
contain (110 for MITM against 20 000 for the DDoS floods), which would make F1 meaningless.
"""
import logging

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)

POOLED = "__pooled__"          # attack-axis entry for all attacks together
CALIBRATION = "__calibration__"  # attack-axis entry for the threshold/FPR rows
ALL_BIN = "all"


def bin_masks(seq_len: np.ndarray,
              bins: tuple[tuple[int, int, str], ...]) -> list[tuple[str, np.ndarray]]:
    """[(bin_name, row_mask)] starting with the unrestricted `all` bin."""
    out = [(ALL_BIN, np.ones(seq_len.shape[0], dtype=bool))]
    for lo, hi, name in bins:
        out.append((name, (seq_len >= lo) & (seq_len <= hi)))
    return out


def calibrate(scores_calib: np.ndarray, quantiles: tuple[float, ...]) -> dict[float, float]:
    """Score thresholds at the given quantiles of the calibration (normal) scores."""
    return {q: float(np.quantile(scores_calib, q)) for q in quantiles}


def _pair_metrics(s_neg: np.ndarray, s_pos: np.ndarray,
                  thresholds: dict[float, float]) -> dict[str, float]:
    """AUROC / AUPRC / partial AUROC plus TPR at each calibrated threshold."""
    y = np.concatenate([np.zeros(s_neg.shape[0]), np.ones(s_pos.shape[0])])
    s = np.concatenate([s_neg, s_pos])
    out = {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        # McClish-standardised partial AUROC over FPR <= 1%: 0.5 is chance, 1.0 perfect.
        "pauroc_fpr1pct": float(roc_auc_score(y, s, max_fpr=0.01)),
    }
    for q, thr in thresholds.items():
        out[f"tpr@q{q}"] = float((s_pos > thr).mean()) if np.isfinite(thr) else float("nan")
    return out


def evaluate(scores: dict[str, np.ndarray], meta: pl.DataFrame,
             masks: dict[str, np.ndarray], quantiles: tuple[float, ...],
             bins: tuple[tuple[int, int, str], ...], min_bin_n: int = 30) -> pl.DataFrame:
    """Long-form metric table: (detector, attack, bin, metric, value).

    `attack` is either a per-file attack label, `__pooled__` for all attacks together, or
    `__calibration__` for the threshold and realised-FPR rows.
    """
    seq_len = meta["seq_len"].to_numpy()
    label = meta["label"].to_numpy()
    neg, pos, calib = masks["eval_neg"], masks["eval_pos"], masks["calib"]
    attacks = sorted(set(label[pos].tolist()))
    rows: list[dict] = []

    def emit(det, attack, bin_name, metric, value):
        rows.append({"detector": det, "attack": attack, "bin": bin_name,
                     "metric": metric, "value": float(value)})

    for det, s in scores.items():
        for bin_name, bmask in bin_masks(seq_len, bins):
            calib_scores = s[calib & bmask]
            neg_scores = s[neg & bmask]
            # A bin-local threshold, so a bin's TPR is read against that bin's own FPR
            # target rather than a global one dominated by whichever bin holds most rows.
            if calib_scores.shape[0] >= min_bin_n:
                thresholds = calibrate(calib_scores, quantiles)
            else:
                thresholds = {q: float("nan") for q in quantiles}

            for q, thr in thresholds.items():
                emit(det, CALIBRATION, bin_name, f"threshold@q{q}", thr)
                realised = ((neg_scores > thr).mean()
                            if neg_scores.shape[0] and np.isfinite(thr) else float("nan"))
                emit(det, CALIBRATION, bin_name, f"fpr_evalneg@q{q}", realised)
            emit(det, CALIBRATION, bin_name, "n_calib", calib_scores.shape[0])
            emit(det, CALIBRATION, bin_name, "n_neg", neg_scores.shape[0])

            for attack in [POOLED] + attacks:
                amask = pos if attack == POOLED else (pos & (label == attack))
                pos_scores = s[amask & bmask]
                emit(det, attack, bin_name, "n_pos", pos_scores.shape[0])
                emit(det, attack, bin_name, "n_neg", neg_scores.shape[0])
                if pos_scores.shape[0] < min_bin_n or neg_scores.shape[0] < min_bin_n:
                    # Not enough on one side to mean anything -- a NaN the reader can trace
                    # back to the n_pos/n_neg rows, rather than a number built on nothing.
                    for metric in ["auroc", "auprc", "pauroc_fpr1pct"] + \
                                  [f"tpr@q{q}" for q in quantiles]:
                        emit(det, attack, bin_name, metric, float("nan"))
                    continue
                for metric, value in _pair_metrics(neg_scores, pos_scores, thresholds).items():
                    emit(det, attack, bin_name, metric, value)

        pooled = [r for r in rows
                  if r["detector"] == det and r["attack"] == POOLED
                  and r["bin"] == ALL_BIN and r["metric"] == "auroc"]
        logger.info(f"  {det}: pooled AUROC {pooled[0]['value']:.4f}")

    return pl.DataFrame(rows)


def agreement_matrix(scores: dict[str, np.ndarray], mask: np.ndarray) -> pl.DataFrame:
    """Spearman correlation between detector scores -- is this suite N views or one?"""
    names = list(scores)
    stacked = np.column_stack([scores[n][mask] for n in names])
    rho = spearmanr(stacked).statistic
    if np.ndim(rho) == 0:      # scipy returns a scalar for exactly two columns
        rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
    return pl.DataFrame({"detector": names,
                         **{n: np.asarray(rho)[:, i] for i, n in enumerate(names)}})


def pivot(metrics: pl.DataFrame, metric: str, bin_name: str = ALL_BIN,
          attacks_only: bool = True) -> pl.DataFrame:
    """(detector x attack) wide view of one metric in one bin, for the heatmaps."""
    df = metrics.filter((pl.col("metric") == metric) & (pl.col("bin") == bin_name))
    if attacks_only:
        df = df.filter(~pl.col("attack").is_in([CALIBRATION]))
    return df.pivot(on="attack", index="detector", values="value")


def pivot_bins(metrics: pl.DataFrame, metric: str, attack: str = POOLED) -> pl.DataFrame:
    """(detector x seq_len bin) wide view of one metric for one attack."""
    return (metrics
            .filter((pl.col("metric") == metric) & (pl.col("attack") == attack))
            .pivot(on="bin", index="detector", values="value"))

"""Figures for gauging the detectors.

Matplotlib PNGs for the writeup, plus one interactive Plotly projection and an HTML report
that stitches everything into a single self-contained file (PNGs inlined as base64, Plotly
from the CDN -- the same pattern VisualizeDistribution.py already uses).

The projection is the point of the exercise: Embedding_Viz.ipynb fits both the scaler and
UMAP on the pooled normals *and* attacks, so the separation it shows is partly the
projection having been told where the attacks are. Here UMAP is fitted on the `train`
normals and everything else is `transform`-ed into that fixed layout.
"""
import base64
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no display in the container
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
from sklearn.decomposition import PCA

from . import evaluation as ev

logger = logging.getLogger(__name__)

ROLE_STYLE = {                 # (colour, draw order) for the score-distribution panels
    "fit": ("#4C78A8", 0),
    "calib": ("#72B7B2", 1),
    "eval_neg": ("#54A24B", 2),
    "eval_pos": ("#E45756", 3),
}
ROLE_LABEL = {"fit": "train (fit)", "calib": "test (calibrate)",
              "eval_neg": "val (eval normal)", "eval_pos": "attacks"}


def _grid(n: int, width: float = 4.2, height: float = 3.2, cols: int = 3):
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows))
    return fig, np.atleast_1d(axes).ravel()


def _heatmap(ax, values: np.ndarray, row_labels, col_labels, title: str,
             vmin: float | None = None, vmax: float | None = None, fmt: str = "{:.2f}"):
    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    finite = values[np.isfinite(values)]
    mid = finite.mean() if finite.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, "--" if not np.isfinite(v) else fmt.format(v),
                    ha="center", va="center", fontsize=6,
                    color="white" if np.isfinite(v) and v < mid else "black")
    return im


def _wide_to_array(df: pl.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    rows = df["detector"].to_list()
    cols = [c for c in df.columns if c != "detector"]
    return df.select(cols).to_numpy().astype(float), rows, cols


# --- 1. score distributions -------------------------------------------------
def plot_score_distributions(scores, masks, metrics, quantiles, out_path):
    """Per detector: where each role's scores sit, with the calibrated thresholds drawn."""
    fig, axes = _grid(len(scores))
    for ax, (det, s) in zip(axes, scores.items()):
        # Robust range: a handful of extreme outliers otherwise squash every distribution
        # into the leftmost bin and the panel says nothing.
        finite = s[np.isfinite(s)]
        lo, hi = np.percentile(finite, [0.2, 99.8])
        bins = np.linspace(lo, hi, 80) if hi > lo else 80
        for role, (colour, _) in sorted(ROLE_STYLE.items(), key=lambda kv: kv[1][1]):
            ax.hist(np.clip(s[masks[role]], lo, hi), bins=bins, density=True, alpha=0.45,
                    color=colour, label=ROLE_LABEL[role])
        for q in quantiles:
            thr = metrics.filter((pl.col("detector") == det) & (pl.col("bin") == ev.ALL_BIN)
                                 & (pl.col("metric") == f"threshold@q{q}"))["value"]
            if thr.len() and np.isfinite(thr[0]):
                ax.axvline(thr[0], color="black", lw=0.8, ls="--")
                ax.text(thr[0], ax.get_ylim()[1] * 0.95, f"q{q}", fontsize=6,
                        rotation=90, va="top", ha="right")
        ax.set_title(det, fontsize=10)
        ax.set_yticks([])
    for ax in axes[len(scores):]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.suptitle("Score distributions by split role (thresholds calibrated on test normals)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- 2. ROC / PR ------------------------------------------------------------
def plot_roc_pr(scores, masks, out_path):
    """Pooled ROC (log FPR, so the operating region is legible) and PR curves."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    neg, pos = masks["eval_neg"], masks["eval_pos"]
    y = np.concatenate([np.zeros(neg.sum()), np.ones(pos.sum())])
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for det, s in scores.items():
        sc = np.concatenate([s[neg], s[pos]])
        fpr, tpr, _ = roc_curve(y, sc)
        ax_roc.plot(np.maximum(fpr, 1e-5), tpr, lw=1.2, label=det)
        prec, rec, _ = precision_recall_curve(y, sc)
        ax_pr.plot(rec, prec, lw=1.2, label=det)
    ax_roc.set_xscale("log")
    ax_roc.set_xlabel("FPR (val normals, log)")
    ax_roc.set_ylabel("TPR (attacks)")
    ax_roc.set_title("ROC, all attacks pooled")
    ax_roc.axvline(0.01, color="grey", lw=0.8, ls=":")
    ax_roc.legend(fontsize=7)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall (prevalence is an artefact of export sizes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- 3. per-attack heatmaps -------------------------------------------------
def plot_attack_heatmaps(metrics, tpr_metric, out_path):
    """The headline figure: detectors x attacks, AUROC and TPR at the 1% FPR threshold.

    Read every row against the `seq_len_only` row at the top -- where a learned detector
    does not beat it, the detection is flow length, not the embedding.
    """
    fig, axes = plt.subplots(2, 1, figsize=(1.0 * 16 + 2, 9))
    for ax, metric, title in [(axes[0], "auroc", "AUROC per attack (vs val normals)"),
                              (axes[1], tpr_metric, f"{tpr_metric} per attack")]:
        values, rows, cols = _wide_to_array(ev.pivot(metrics, metric))
        im = _heatmap(ax, values, rows, cols, title, vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- 4. length-stratified ---------------------------------------------------
def plot_stratified(metrics, best_detector, out_path):
    """Detectors x seq_len bin for the pooled attacks, and attacks x bin for one detector.

    Attack windows are scored only against normals in the same bin, so these cells are the
    ones that survive the length confound.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    values, rows, cols = _wide_to_array(ev.pivot_bins(metrics, "auroc", ev.POOLED))
    im = _heatmap(axes[0], values, rows, cols,
                  "AUROC by seq_len bin (all attacks pooled, length-matched normals)",
                  vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=axes[0], fraction=0.04)

    per_attack = (metrics
                  .filter((pl.col("metric") == "auroc") & (pl.col("detector") == best_detector)
                          & (pl.col("attack") != ev.CALIBRATION))
                  .pivot(on="bin", index="attack", values="value"))
    vals = per_attack.select([c for c in per_attack.columns if c != "attack"]).to_numpy().astype(float)
    im = _heatmap(axes[1], vals, per_attack["attack"].to_list(),
                  [c for c in per_attack.columns if c != "attack"],
                  f"AUROC by attack x seq_len bin -- {best_detector}", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- 5. detector agreement --------------------------------------------------
def plot_agreement(agree: pl.DataFrame, out_path):
    values, rows, cols = _wide_to_array(agree)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Spearman correlation between detector scores (eval rows)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- 6. score vs seq_len ----------------------------------------------------
def plot_score_vs_seqlen(scores, meta, masks, bins, out_path):
    """Makes the confound visible: does the score track flow length inside each role?"""
    seq_len = meta["seq_len"].to_numpy()
    bin_defs = ev.bin_masks(seq_len, bins)[1:]     # drop the unrestricted `all` bin
    fig, axes = _grid(len(scores), width=4.6, height=3.4)
    for ax, (det, s) in zip(axes, scores.items()):
        positions, data, colours = [], [], []
        for i, (bin_name, bmask) in enumerate(bin_defs):
            for j, role in enumerate(["eval_neg", "eval_pos"]):
                vals = s[masks[role] & bmask]
                if vals.size < 5:
                    continue
                positions.append(i * 3 + j)
                data.append(vals)
                colours.append(ROLE_STYLE[role][0])
        box = ax.boxplot(data, positions=positions, widths=0.8, showfliers=False,
                         patch_artist=True)
        for patch, colour in zip(box["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(bin_defs))],
                      [b for b, _ in bin_defs], fontsize=7)
        ax.set_title(det, fontsize=10)
    for ax in axes[len(scores):]:
        ax.axis("off")
    fig.suptitle("Score vs seq_len bin -- left box val normals, right box attacks")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- 7. calibration ---------------------------------------------------------
def plot_calibration(metrics, quantiles, out_path):
    """Nominal FPR (from the test quantile) against what the val normals actually give."""
    fig, ax = plt.subplots(figsize=(6, 6))
    nominal = [1 - q for q in quantiles]
    for det in metrics["detector"].unique(maintain_order=True):
        realised = [metrics.filter((pl.col("detector") == det)
                                   & (pl.col("bin") == ev.ALL_BIN)
                                   & (pl.col("metric") == f"fpr_evalneg@q{q}"))["value"][0]
                    for q in quantiles]
        ax.plot(nominal, realised, marker="o", lw=1.2, label=det)
    lims = [min(nominal) / 2, max(nominal) * 2]
    ax.plot(lims, lims, color="black", ls="--", lw=0.8, label="perfect calibration")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nominal FPR (1 - quantile of test normals)")
    ax.set_ylabel("realised FPR on val normals")
    ax.set_title("Threshold transfer test -> val")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- 8. normal-fitted projection -------------------------------------------
def plot_projection(X, meta, masks, score, score_name, out_html, out_png,
                    n_projection=50_000, n_neighbors=100, min_dist=0.05, seed=0):
    """UMAP fitted on the fit split alone, everything else transformed into it.

    A PCA-fit-on-train panel rides alongside as a deterministic cross-check: if the two
    disagree about the structure, neither is telling you much. Falls back to PCA only if
    umap-learn turns out not to be installed.
    """
    from .data import subsample, take_rows
    idx = subsample(meta, max(1, n_projection // meta["label"].n_unique()), seed)
    Xs, meta_s, score_s = X[idx], take_rows(meta, idx), score[idx]

    embeddings = {}
    pca = PCA(n_components=2, random_state=seed).fit(X[masks["fit"]])
    embeddings["PCA (fit on train)"] = pca.transform(Xs)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                            metric="cosine", random_state=seed).fit(X[masks["fit"]])
        embeddings["UMAP (fit on train)"] = reducer.transform(Xs)
    except ImportError:
        logger.warning("umap-learn not installed -- projection falls back to PCA only")

    labels = meta_s["label"].to_numpy()
    fig, axes = plt.subplots(len(embeddings), 2,
                             figsize=(13, 6 * len(embeddings)), squeeze=False)
    for row, (name, Z) in enumerate(embeddings.items()):
        ax = axes[row][0]
        for i, cat in enumerate(sorted(set(labels.tolist()))):
            m = labels == cat
            ax.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.5, label=cat,
                       color=plt.cm.tab20(i % 20))
        ax.set_title(f"{name} -- by set")
        ax.set_xticks([]), ax.set_yticks([])
        ax.legend(markerscale=4, fontsize=5, loc="best")

        ax = axes[row][1]
        lo, hi = np.percentile(score_s, [1, 99])
        sc = ax.scatter(Z[:, 0], Z[:, 1], s=2, alpha=0.6,
                        c=np.clip(score_s, lo, hi), cmap="magma")
        ax.set_title(f"{name} -- by {score_name} score")
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # Interactive twin of the left column, so individual attack sets can be toggled.
    name, Z = list(embeddings.items())[-1]
    go_fig = go.Figure()
    for cat in sorted(set(labels.tolist())):
        m = labels == cat
        go_fig.add_trace(go.Scattergl(x=Z[m, 0], y=Z[m, 1], mode="markers", name=str(cat),
                                      marker=dict(size=3, opacity=0.6),
                                      hovertemplate=f"{cat}<extra></extra>"))
    go_fig.update_layout(
        title=f"{name} -- normals-only projection, attacks transformed in",
        width=1000, height=800,
        legend=dict(itemsizing="constant", itemclick="toggle", itemdoubleclick="toggleothers"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
        template="plotly_white",
    )
    go_fig.write_html(out_html, include_plotlyjs="cdn")
    return out_png, go_fig


# --- report -----------------------------------------------------------------
def write_report(out_path, sections: list[tuple[str, str, object]]):
    """Single self-contained HTML: PNGs inlined as base64, Plotly figures as divs."""
    parts = ["<html><head><meta charset='utf-8'><title>Embedding AD report</title>",
             "<style>body{font-family:sans-serif;max-width:1400px;margin:2em auto;}"
             "img{max-width:100%;}h2{border-bottom:1px solid #ccc;}</style></head><body>"]
    first_plotly = True
    for title, kind, payload in sections:
        parts.append(f"<h2>{title}</h2>")
        if kind == "png":
            b64 = base64.b64encode(Path(payload).read_bytes()).decode()
            parts.append(f"<img src='data:image/png;base64,{b64}'/>")
        elif kind == "plotly":
            parts.append(payload.to_html(full_html=False,
                                         include_plotlyjs="cdn" if first_plotly else False))
            first_plotly = False
        elif kind == "html":
            parts.append(payload)
    parts.append("</body></html>")
    Path(out_path).write_text("\n".join(parts))
    return out_path


def table_html(df: pl.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Small polars frame as a plain HTML table for the report."""
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = []
    for row in df.iter_rows():
        cells = "".join(
            f"<td>{float_fmt.format(v) if isinstance(v, float) else v}</td>" for v in row)
        body.append(f"<tr>{cells}</tr>")
    return (f"<table border='1' cellpadding='4' cellspacing='0'>"
            f"<tr>{head}</tr>{''.join(body)}</table>")

"""
UMAP projection of exported flow/packet embeddings.

What is compared to what: the reference is the **test split** (held-out normal
traffic, monitored during training), and everything else plotted is a per-capture
attack set. train is excluded (the models fit it) and **val is excluded on purpose**
-- it is the final held-out set, spent once in a single final evaluation.

The UMAP manifold is fit on the test normals **alone**; attacks are transformed into
it. Fitting on the union would let a large attack set help define the space it is
then judged against and appear separated by construction.

Reads a directory of <label>.npy embedding matrices plus the metadata.parquet that
SequenceEmbeddingAD writes alongside them, and renders three views of one projection:

  umap_overview.png        normal vs attack, the headline claim
  umap_small_multiples.png one panel per attack, against a grey normal background
  umap_seq_len.png         the same points coloured by flow length

The third figure is not decoration. Normal traffic here has a median flow length of
64 packets while most attack sets are 1-2, and the sequence encoder's length head
scores ~0.978 -- so length is provably in the bottleneck vector. If the attack
clusters coincide with the short-flow region of the length plot, the separation is
length, not attack behaviour. Read figure 3 before believing figure 1.

Seventeen categories on one scatter is unreadable and not colourblind-safe, which is
why the per-category view is small multiples rather than a 17-colour legend.

Run from the repo root:
    python -m RawByteTrafficModelling.AnomalyDetection.embedding_viz
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
import polars as pl
import numpy as np
import umap

DATA_DIR = Path("RawByteTrafficModelling/AnomalyDetection/Outputs/Embeddings/"
                "SequenceEmbeddings_SeqAE_IIoTset_d128_Mamba_s512")
SEED = 42

# --- What is compared to what ------------------------------------------------
# Reference ("normal") is the TEST split: held out from packet- and sequence-AE
# training, monitored during it. Attacks are the per-capture attack sets.
#
# train is excluded because the models fit it, so it cannot say anything about
# generalisation. val is excluded deliberately: it is the final held-out set and is
# spent once, in a single final evaluation -- do not add it here to "check".
NORMAL_SET = "test"
EXCLUDED_SETS = {"train", "val"}

NORMAL_CAP = 8000       # the background cloud, so sampled denser than any attack
ATTACK_CAP = 3000

# The UMAP manifold is fit on NORMAL_SET alone and the attacks are *transformed*
# into it. Fitting on the union would let the attacks help define the space they
# are then judged against, so a large attack set could carve out its own region and
# look separated by construction. Fit-on-normal makes "far from the cloud" mean
# "off the normal manifold", which is the actual anomaly-detection question.
UMAP_KWARGS = dict(n_neighbors=50, min_dist=0.1, metric="cosine", random_state=SEED)

# Palette: slots 1 and 2 of the reference categorical set, in fixed order (the two
# most-separated hues), plus its single-hue blue sequential ramp. No invented colours.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
CONTEXT_GREY = "#c8c7c0"       # background cloud: context, not a series
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_RAMP = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)

rng = np.random.default_rng(SEED)

# ---- load ------------------------------------------------------------------
# Row i of metadata.parquet describes row i of the stacked matrix, in sorted-label
# order -- the contract SequenceEmbeddingAD writes to. Verified per label below.
meta = pl.read_parquet(DATA_DIR / "metadata.parquet")

X_parts, labels, seq_lens = [], [], []
for f in sorted(DATA_DIR.glob("*.npy")):
    # "umap_*" are this script's own outputs written into the same directory; without
    # this they get globbed back in as if they were an embedding set.
    if f.stem in EXCLUDED_SETS or f.stem.startswith("umap_"):
        continue
    arr = np.load(f)
    rows = meta.filter(pl.col("label") == f.stem)
    assert rows.height == arr.shape[0], (
        f"{f.stem}: {arr.shape[0]} embeddings but {rows.height} metadata rows -- "
        f"the .npy files and metadata.parquet are out of sync")

    n = NORMAL_CAP if f.stem == NORMAL_SET else ATTACK_CAP
    take = rng.choice(arr.shape[0], min(n, arr.shape[0]), replace=False)

    X_parts.append(arr[take])
    labels += [f.stem] * take.shape[0]
    seq_lens.append(rows["seq_len"].to_numpy()[take])

X = np.vstack(X_parts).astype(np.float32)
labels = np.array(labels)
seq_lens = np.concatenate(seq_lens)
is_normal = labels == NORMAL_SET
attack_names = [c for c in sorted(set(labels)) if c != NORMAL_SET]
print(f"excluded from this view: {sorted(EXCLUDED_SETS)}")
print(f"{X.shape[0]} points, {X.shape[1]} dims, "
      f"{is_normal.sum()} normal ({NORMAL_SET}) / {(~is_normal).sum()} attack, "
      f"{len(attack_names)} attack sets")

# L2-normalise, then standardise: cosine geometry is what UMAP is asked for below.
# The scaler is fit on the normal reference only, for the same reason the UMAP is --
# the attacks must not influence the space they are measured in.
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
scaler = StandardScaler().fit(X[is_normal])
X = scaler.transform(X)

# ---- project ---------------------------------------------------------------
# Fit on the normal reference, then transform the attacks into that manifold.
print(f"fitting UMAP on {NORMAL_SET} only ({is_normal.sum()} points) {UMAP_KWARGS} ...")
reducer = umap.UMAP(**UMAP_KWARGS).fit(X[is_normal])
Z = np.empty((X.shape[0], 2), dtype=np.float32)
Z[is_normal] = reducer.embedding_
print(f"transforming {(~is_normal).sum()} attack points into it ...")
Z[~is_normal] = reducer.transform(X[~is_normal])
np.save(DATA_DIR / "umap_coords.npy", Z)   # so replotting costs nothing
print("UMAP done")

lim = [(Z[:, 0].min(), Z[:, 0].max()), (Z[:, 1].min(), Z[:, 1].max())]


def style(ax, title, subtitle=None):
    """Title above subtitle, both left-aligned, neither overlapping the other.

    set_title sits just above the axes, so a subtitle at y~1.0 lands on top of it.
    Reserving space with pad and placing the subtitle just above the axes puts them
    in separate bands.
    """
    ax.set_title(title, fontsize=10, color=INK, loc="left",
                 pad=20 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=8,
                color=INK_MUTED, va="bottom", ha="left")
    ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#e0dfd8")


# ---- 1. overview: normal vs attack -----------------------------------------
PROVENANCE = (f"UMAP fit on {NORMAL_SET} normals only, attacks transformed in | "
              f"512-dim SeqAE_IIoTset_d128_Mamba_s512 bottleneck, cosine")

fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(Z[is_normal, 0], Z[is_normal, 1], s=4, alpha=0.35, color=BLUE,
           linewidths=0, label=f"normal — {NORMAL_SET} split ({is_normal.sum()})")
ax.scatter(Z[~is_normal, 0], Z[~is_normal, 1], s=4, alpha=0.35, color=ORANGE,
           linewidths=0, label=f"attack — 14 captures ({(~is_normal).sum()})")
style(ax, f"Normal ({NORMAL_SET} split) vs attack captures", PROVENANCE)
ax.legend(markerscale=4, fontsize=9, loc="best", frameon=False)
fig.tight_layout()
fig.savefig(DATA_DIR / "umap_overview.png", dpi=200)
print("wrote umap_overview.png")

# ---- 2. small multiples: one attack at a time -------------------------------
ncol = 4
nrow = int(np.ceil(len(attack_names) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow))
for ax, name in zip(axes.ravel(), attack_names):
    m = labels == name
    ax.scatter(Z[is_normal, 0], Z[is_normal, 1], s=3, alpha=0.25,
               color=CONTEXT_GREY, linewidths=0)
    ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.6, color=ORANGE, linewidths=0)
    med = int(np.median(seq_lens[m]))
    style(ax, name.replace("_", " "), f"{m.sum()} windows, median flow {med} packets")
for ax in axes.ravel()[len(attack_names):]:
    ax.axis("off")
fig.suptitle(f"Each attack capture (orange) against the {NORMAL_SET}-split "
             f"normal background (grey)\n{PROVENANCE}",
             fontsize=11, color=INK, x=0.01, ha="left", va="top")
# Leave the suptitle its own band so it cannot land on the first row's titles.
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(DATA_DIR / "umap_small_multiples.png", dpi=200)
print("wrote umap_small_multiples.png")

# ---- 3. the confound: same projection, coloured by flow length --------------
fig, ax = plt.subplots(figsize=(9, 8))
order = np.argsort(seq_lens)          # long flows drawn last, on top
sc = ax.scatter(Z[order, 0], Z[order, 1], s=4, alpha=0.5, c=seq_lens[order],
                cmap=BLUE_RAMP, linewidths=0)
style(ax, f"Same points ({NORMAL_SET} + attacks), coloured by flow length",
      "If the attack clusters sit in the light region, the separation is length, "
      "not behaviour")
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("packets in window", fontsize=9, color=INK_MUTED)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(DATA_DIR / "umap_seq_len.png", dpi=200)
print("wrote umap_seq_len.png")

# ---- the same question, as numbers ------------------------------------------
# A picture can suggest separation; this states it. Per set: median flow length and
# the share of its windows that are single-packet.
print(f"\nreference = {NORMAL_SET} split; everything below it is an attack capture")
print("\nset                              n   median_len  %single_packet")
for name in [NORMAL_SET] + attack_names:
    m = labels == name
    print(f"{name[:30]:30s} {m.sum():6d} {int(np.median(seq_lens[m])):8d} "
          f"{100 * (seq_lens[m] == 1).mean():13.1f}")

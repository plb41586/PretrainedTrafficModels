"""Load, split and preprocess an embedding export directory.

The contract with SequenceEmbeddingAD.py: one `.npy` per set, plus a `metadata.parquet`
whose row i describes row i of the `.npy` files stacked in `sorted(glob("*.npy"))` order
(that script builds the metadata with `sorted(embeddings, key=lambda n: f"{n}.npy")`, which
is the same order). Every number this suite produces rests on that alignment, so
`load_embedding_dir` checks it run-length by run-length rather than trusting it.

Split roles follow the project's naming: `train` is fitted on, `test` was monitored during
training and is therefore already spent (it is where thresholds get calibrated), and `val`
is the final held-out set used as the evaluation negatives.
"""
from pathlib import Path
import logging

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_embedding_dir(data_dir: str) -> tuple[np.ndarray, pl.DataFrame]:
    """Stacked (N, D) embeddings plus the row-aligned metadata frame.

    Labels come from `metadata.parquet`, never re-derived from the filenames -- but the
    file order is used to *verify* the metadata, since a mismatch would silently attribute
    attack rows to normal splits and quietly invert every result.

    Only files whose stem is a label in metadata.parquet are read. Downstream tools write
    their own .npy artefacts into this directory (embedding_viz caches umap_coords.npy),
    and those must not be stacked in as if they were an exported set.
    """
    data_dir = Path(data_dir)
    known = set(pl.read_parquet(data_dir / "metadata.parquet")["label"].unique().to_list())
    files = [f for f in sorted(data_dir.glob("*.npy")) if f.stem in known]
    if not files:
        raise FileNotFoundError(f"no .npy files in {data_dir} matching a metadata label")
    skipped = sorted(f.stem for f in data_dir.glob("*.npy") if f.stem not in known)
    if skipped:
        logger.info(f"Ignoring non-embedding .npy in {data_dir}: {skipped}")

    parts, expected = [], []
    for f in files:
        arr = np.load(f)
        parts.append(arr.astype(np.float32, copy=False))
        expected.append((f.stem, arr.shape[0]))
    X = np.vstack(parts)

    meta = pl.read_parquet(data_dir / "metadata.parquet")
    if meta.height != X.shape[0]:
        raise ValueError(f"metadata has {meta.height} rows but the .npy files stack to "
                         f"{X.shape[0]} -- the export and the metadata are out of sync")

    labels = meta["label"].to_numpy()
    offset = 0
    for stem, n in expected:
        block = labels[offset:offset + n]
        if not (block == stem).all():
            raise ValueError(
                f"metadata rows {offset}:{offset + n} should all be label '{stem}' "
                f"(from {stem}.npy) but hold {sorted(set(block))} -- the metadata is not in "
                f"sorted-filename order and every downstream number would be misattributed")
        offset += n

    logger.info(f"Loaded {X.shape[0]} embeddings of dim {X.shape[1]} from {len(files)} sets")
    return X, meta


def subsample(meta: pl.DataFrame, n_max: int, seed: int) -> np.ndarray:
    """Row indices capping each label at `n_max`, evenly spaced (deterministic, spans the set).

    Even spacing rather than a random draw for the same reason `SequenceEmbeddingAD.pick_windows`
    does it: the exports are in flow order, so a prefix or a random draw both distort which
    flows survive, while a linspace keeps the whole span of the split.
    """
    del seed  # kept in the signature so callers can switch to a random draw without churn
    keep = []
    for label in meta["label"].unique(maintain_order=True):
        idx = np.flatnonzero((meta["label"] == label).to_numpy())
        if idx.size > n_max:
            idx = idx[np.linspace(0, idx.size - 1, n_max).astype(np.int64)]
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def take_rows(df: pl.DataFrame, idx: np.ndarray) -> pl.DataFrame:
    """Row subset by integer index. `subsample` returns sorted indices, so a boolean filter
    is equivalent to a gather and avoids relying on polars' int-array indexing semantics."""
    mask = np.zeros(df.height, dtype=bool)
    mask[idx] = True
    return df.filter(pl.Series(mask))


def split_roles(meta: pl.DataFrame, fit_label: str, calib_label: str,
                eval_neg_label: str, attack_set: str = "attack",
                calib_frac: float = 0.5, seed: int = 0) -> dict[str, np.ndarray]:
    """Boolean row masks for the four roles, checked to be non-empty and disjoint.

    If calib_label == eval_neg_label, that one set plays both roles and is split
    deterministically into two disjoint halves (calib_frac to calibration). That is
    what keeps thresholds from being calibrated on the very rows they are then
    evaluated against, and it is how the final held-out split stays unspent while
    still getting an honest false-positive rate.
    """
    label = meta["label"].to_numpy()
    set_col = meta["set"].to_numpy()
    masks = {
        "fit": label == fit_label,
        "calib": label == calib_label,
        "eval_neg": label == eval_neg_label,
        "eval_pos": set_col == attack_set,
    }

    if calib_label == eval_neg_label:
        idx = np.where(label == calib_label)[0]
        shuffled = np.random.default_rng(seed).permutation(idx)
        n_calib = int(round(calib_frac * idx.size))
        calib_mask = np.zeros_like(masks["calib"])
        eval_mask = np.zeros_like(masks["eval_neg"])
        calib_mask[shuffled[:n_calib]] = True
        eval_mask[shuffled[n_calib:]] = True
        masks["calib"], masks["eval_neg"] = calib_mask, eval_mask
        logger.info(f"'{calib_label}' plays both calib and eval_neg: split "
                    f"{n_calib}/{idx.size - n_calib} (disjoint, seed {seed})")
    for name, m in masks.items():
        if not m.any():
            raise ValueError(f"role '{name}' selects no rows -- check the *_LABEL config "
                             f"against the exported labels {sorted(set(label))}")
    stacked = np.stack(list(masks.values()))
    if (stacked.sum(axis=0) > 1).any():
        raise ValueError("the four role masks overlap; a row cannot be both fitted on and "
                         "evaluated against")
    logger.info("Roles: " + ", ".join(f"{n}={int(m.sum())}" for n, m in masks.items()))
    return masks


class Preprocessor:
    """Scaling (and optional PCA) fitted on the fit split only.

    `l2_standardize` reproduces what Embedding_Viz.ipynb does to the embeddings before UMAP
    (L2-normalise, then standardise), so the detectors see the same geometry the promising
    UMAP picture came from -- the difference is that here the statistics come from `train`
    alone instead of the pooled normals+attacks.
    """

    MODES = ("l2_standardize", "standardize", "raw")

    def __init__(self, mode: str = "l2_standardize", pca_components: int | float | None = None,
                 seed: int = 0):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.pca_components = pca_components
        self.seed = seed
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None

    def _l2(self, X: np.ndarray) -> np.ndarray:
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    def fit(self, X_fit: np.ndarray) -> "Preprocessor":
        Z = self._l2(X_fit) if self.mode == "l2_standardize" else X_fit
        if self.mode != "raw":
            self.scaler = StandardScaler().fit(Z)
            Z = self.scaler.transform(Z)
        if self.pca_components:
            # svd_solver="full" explicitly: a fractional n_components (keep 95% variance)
            # is only accepted by the full solver.
            self.pca = PCA(n_components=self.pca_components, svd_solver="full",
                           random_state=self.seed).fit(Z)
            logger.info(f"PCA fitted on the fit split: {Z.shape[1]} -> "
                        f"{self.pca.n_components_} components, "
                        f"{self.pca.explained_variance_ratio_.sum():.3f} variance kept")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._l2(X) if self.mode == "l2_standardize" else X
        if self.scaler is not None:
            Z = self.scaler.transform(Z)
        if self.pca is not None:
            Z = self.pca.transform(Z)
        return np.ascontiguousarray(Z, dtype=np.float32)

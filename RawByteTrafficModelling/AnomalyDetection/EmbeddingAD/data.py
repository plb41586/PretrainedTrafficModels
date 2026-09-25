"""Load, split and preprocess an embedding export directory.

The contract with SequenceEmbeddingAD.py: one `.npy` per set, plus a `metadata.parquet`
whose row i describes row i of the `.npy` files stacked in `sorted(glob("*.npy"))` order
(that script builds the metadata with `sorted(embeddings, key=lambda n: f"{n}.npy")`, which
is the same order). Every number this suite produces rests on that alignment, so
`load_embedding_dir` checks it run-length by run-length rather than trusting it.

Which exported sets play which role is the caller's DATA_ROLES map (RunConfig.DataRoles);
only the sets it names are read from disk.
"""
from pathlib import Path
import logging

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from RawByteTrafficModelling.PreTraining.RunConfig import ATTACK, NORMAL, DataRoles

logger = logging.getLogger(__name__)

ATTACK_SET = "attack"   # the `set` column value SequenceEmbeddingAD writes for attack rows


def embedding_sections(data_dir: str) -> dict[str, str]:
    """Every exported set in an embedding directory -> NORMAL | ATTACK, for `DataRoles`.

    Reads only the `set`/`label` columns of `metadata.parquet` -- the index of what the
    directory holds -- not any embedding.
    """
    pairs = (pl.read_parquet(Path(data_dir) / "metadata.parquet", columns=["set", "label"])
             .unique(maintain_order=True))
    return {label: ATTACK if s == ATTACK_SET else NORMAL
            for s, label in pairs.iter_rows()}


def load_embedding_dir(data_dir: str, roles: DataRoles) -> tuple[np.ndarray, pl.DataFrame]:
    """Stacked (N, D) embeddings plus the row-aligned metadata frame, for the mapped sets only.

    Only the `.npy` of a set that some role maps is opened, and each open is logged through
    `roles.touch`; the metadata is filtered to the same sets (order kept). An unmapped
    split is therefore never read, rather than read and dropped.

    Labels come from `metadata.parquet`, never re-derived from the filenames -- but the
    file order is used to *verify* the metadata, since a mismatch would silently attribute
    attack rows to normal splits and quietly invert every result.
    """
    data_dir = Path(data_dir)
    wanted = {s: role for role, secs in roles.roles.items() for s in secs}
    # Same order SequenceEmbeddingAD writes the metadata in: sorted by file name.
    files = [data_dir / f"{label}.npy" for label in sorted(wanted, key=lambda n: f"{n}.npy")]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"DATA_ROLES maps sets with no file in {data_dir}: {missing}")

    parts, expected = [], []
    for f in files:
        arr = np.load(f)
        roles.touch(wanted[f.stem], f.stem, str(f), rows=arr.shape[0])
        parts.append(arr.astype(np.float32, copy=False))
        expected.append((f.stem, arr.shape[0]))
    X = np.vstack(parts)

    meta = (pl.read_parquet(data_dir / "metadata.parquet")
            .filter(pl.col("label").is_in(list(wanted))))
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


def split_roles(meta: pl.DataFrame, roles: DataRoles,
                required: tuple[str, ...] = ("fit", "calib", "eval_neg", "eval_pos"),
                ) -> dict[str, np.ndarray]:
    """Boolean row masks, one per role in the map, checked to be disjoint.

    Every `required` role must select rows; any other role (e.g. `drift`) may be mapped
    to nothing, and is then absent from the result. A section serves exactly one role --
    there is no splitting of one set into two roles.
    """
    label = meta["label"]
    masks = {}
    for role, sections in roles.roles.items():
        m = label.is_in(list(sections)).to_numpy()
        if not m.any():
            if role in required:
                raise ValueError(f"role '{role}' selects no rows -- check DATA_ROLES "
                                 f"against the exported labels {sorted(set(label.to_list()))}")
            continue
        masks[role] = m
    stacked = np.stack(list(masks.values()))
    if (stacked.sum(axis=0) > 1).any():
        raise ValueError("role masks overlap; a row cannot be both fitted on and "
                         "evaluated against")
    logger.info("Roles: " + ", ".join(f"{n}={int(m.sum())}" for n, m in masks.items()))
    return masks


class Preprocessor:
    """Scaling (and optional PCA) fitted on the fit split only.

    `l2_standardize` reproduces what Embedding_Viz.ipynb does to the embeddings before UMAP
    (L2-normalise, then standardise), so the detectors see the same geometry the promising
    UMAP picture came from -- the difference is that here the statistics come from the fit
    role alone instead of the pooled normals+attacks.
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

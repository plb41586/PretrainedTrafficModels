"""One-class anomaly detectors over embedding vectors.

Every detector is fitted on the `train` normals alone and exposes the same two calls, so
the driver never special-cases one:

    det.fit(X_fit, meta_fit)
    det.score(X, meta) -> (N,) float64, **higher = more anomalous**

`DETECTORS` is the registry the driver iterates -- same idea as the `BACKBONES` factory in
ModelDefinitions.py, adding a detector is one entry. `seq_len_only` is deliberately in the
registry rather than off to the side: the attack exports are dominated by single-packet
windows while the normal splits average ~35 packets, so a detector that does not beat the
flow-length baseline on a given attack class has told us nothing about the embedding.
"""
from typing import Callable, Protocol
import logging

import numpy as np
import polars as pl
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

SCORE_CHUNK = 20_000   # rows per scoring block; bounds peak memory on the 200k+ eval side


class Detector(Protocol):
    name: str

    def fit(self, X: np.ndarray, meta: pl.DataFrame) -> None: ...

    def score(self, X: np.ndarray, meta: pl.DataFrame) -> np.ndarray: ...


def _chunks(n: int, size: int = SCORE_CHUNK):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def _blockwise(fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> np.ndarray:
    """Apply a scoring function in row blocks and concatenate."""
    return np.concatenate([fn(X[sl]) for sl in _chunks(X.shape[0])])


class SeqLenBaseline:
    """Flow length as the only signal. The bar every learned detector has to clear."""

    name = "seq_len_only"

    def fit(self, X, meta):
        pass

    def score(self, X, meta):
        return -meta["seq_len"].to_numpy().astype(np.float64)


class MahalanobisDetector:
    """Squared Mahalanobis distance to the fit-split mean, Ledoit-Wolf shrunk covariance.

    Shrinkage rather than the empirical covariance because 384 dimensions against 20k
    samples leaves the empirical estimate badly conditioned.
    """

    name = "mahalanobis"

    def __init__(self):
        self.cov = LedoitWolf()

    def fit(self, X, meta):
        self.cov.fit(X.astype(np.float64))
        logger.info(f"  {self.name}: shrinkage {self.cov.shrinkage_:.4f}")

    def score(self, X, meta):
        return _blockwise(lambda B: self.cov.mahalanobis(B.astype(np.float64)), X)


class PCAReconstruction:
    """Residual norm outside the top-k principal subspace of the fit split.

    The embedding-space analogue of the autoencoder story: how much of a vector lives in
    directions the normal traffic never uses.
    """

    name = "pca_recon"

    def __init__(self, n_components: int | float = 0.9):
        self.n_components = n_components
        self.pca: PCA | None = None

    def fit(self, X, meta):
        # svd_solver="full": a fractional n_components (keep 90% variance) needs it.
        self.pca = PCA(n_components=self.n_components, svd_solver="full", random_state=0).fit(X)
        logger.info(f"  {self.name}: {self.pca.n_components_} components, "
                    f"{self.pca.explained_variance_ratio_.sum():.3f} variance kept")

    def _residual(self, B: np.ndarray) -> np.ndarray:
        recon = self.pca.inverse_transform(self.pca.transform(B))
        return np.linalg.norm(B - recon, axis=1).astype(np.float64)

    def score(self, X, meta):
        return _blockwise(self._residual, X)


class KNNDetector:
    """Mean distance to the k nearest fit-split points.

    Cosine by default to match the metric Embedding_Viz.ipynb hands UMAP, so this is the
    closest numerical stand-in for "is this point off in its own region of the plot".
    """

    def __init__(self, k: int = 20, metric: str = "cosine"):
        self.k = k
        self.metric = metric
        self.name = f"knn_{k}"
        self.nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="brute", n_jobs=-1)

    def fit(self, X, meta):
        self.nn.fit(X)

    def score(self, X, meta):
        return _blockwise(lambda B: self.nn.kneighbors(B, return_distance=True)[0]
                          .mean(axis=1).astype(np.float64), X)


class LOFDetector:
    """Local Outlier Factor in novelty mode -- local density, so multi-modal normals are fine.

    The normal traffic is a mixture of per-protocol clusters of very different tightness,
    which is exactly the case a single global density gets wrong.
    """

    name = "lof"

    def __init__(self, n_neighbors: int = 20, metric: str = "cosine"):
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric=metric,
                                      novelty=True, n_jobs=-1)

    def fit(self, X, meta):
        self.lof.fit(X)

    def score(self, X, meta):
        # score_samples is the *negated* LOF: higher means more normal.
        return _blockwise(lambda B: -self.lof.score_samples(B).astype(np.float64), X)


class IForestDetector:
    """Isolation Forest -- axis-aligned partitioning baseline, cheap at this scale."""

    name = "iforest"

    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.forest = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1)

    def fit(self, X, meta):
        self.forest.fit(X)

    def score(self, X, meta):
        return _blockwise(lambda B: -self.forest.score_samples(B).astype(np.float64), X)


class GMMDetector:
    """Negative log-likelihood under a diagonal Gaussian mixture, n_components picked by BIC.

    Multi-modal like LOF but without keeping the whole fit split around at scoring time.
    """

    name = "gmm"

    def __init__(self, candidates: tuple[int, ...] = (1, 4, 8, 16, 32), seed: int = 0):
        self.candidates = candidates
        self.seed = seed
        self.gmm: GaussianMixture | None = None

    def fit(self, X, meta):
        best = None
        for n in self.candidates:
            gmm = GaussianMixture(n_components=n, covariance_type="diag",
                                  random_state=self.seed, reg_covar=1e-4).fit(X)
            bic = gmm.bic(X)
            logger.info(f"  {self.name}: n_components={n} BIC={bic:.0f}")
            if best is None or bic < best[0]:
                best = (bic, gmm)
        self.gmm = best[1]
        logger.info(f"  {self.name}: selected n_components={self.gmm.n_components}")

    def score(self, X, meta):
        return _blockwise(lambda B: -self.gmm.score_samples(B).astype(np.float64), X)


class OCSVMDetector:
    """RBF one-class SVM on a subsample of the fit split.

    Subsampled because the fit is quadratic in the number of points; the slow member of the
    suite, so the driver keeps it behind ENABLE_SLOW.
    """

    name = "ocsvm"

    def __init__(self, max_fit: int = 10_000, nu: float = 0.05, seed: int = 0):
        self.max_fit = max_fit
        self.seed = seed
        self.svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)

    def fit(self, X, meta):
        if X.shape[0] > self.max_fit:
            idx = np.linspace(0, X.shape[0] - 1, self.max_fit).astype(np.int64)
            X = X[idx]
        self.svm.fit(X)
        logger.info(f"  {self.name}: fitted on {X.shape[0]} points, "
                    f"{self.svm.support_vectors_.shape[0]} support vectors")

    def score(self, X, meta):
        return _blockwise(lambda B: -self.svm.decision_function(B).astype(np.float64), X)


# Registry the driver iterates. Order is the order they appear in every table and figure,
# so the baseline sits first and everything below it is read against it.
DETECTORS: dict[str, Callable[[], Detector]] = {
    "seq_len_only": SeqLenBaseline,
    "mahalanobis": MahalanobisDetector,
    "pca_recon": PCAReconstruction,
    "knn_1": lambda: KNNDetector(k=1),
    "knn_20": lambda: KNNDetector(k=20),
    "lof": LOFDetector,
    "iforest": IForestDetector,
    "gmm": GMMDetector,
    "ocsvm": OCSVMDetector,
}

# Members whose fit or scoring cost is out of proportion to the rest.
SLOW_DETECTORS = {"ocsvm"}
